const std = @import("std");
const assert = std.debug.assert;

// File d'attente intrusive SPSC (un producteur, un consommateur) ; `T` doit avoir un champ `next: ?*T`.
/// An intrusive queue implementation. The type T must have a field
/// "next" of type `?*T`.
///
/// For those unaware, an intrusive variant of a data structure is one in which
/// the data type in the list has the pointer to the next element, rather
/// than a higher level "node" or "container" type. The primary benefit
/// of this (and the reason we implement this) is that it defers all memory
/// management to the caller: the data structure implementation doesn't need
/// to allocate "nodes" to contain each element. Instead, the caller provides
/// the element and how its allocated is up to them.
pub fn SPSC(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Head is the front of the queue and tail is the back of the queue.
        head: ?*T = null,
        tail: ?*T = null,

        // Ajoute `v` à l'arrière de la file.
        /// Enqueue a new element to the back of the queue.
        pub fn push(self: *Self, v: *T) void {
            assert(v.next == null);

            if (self.tail) |tail| {
                // If we have elements in the queue, then we add a new tail.
                tail.next = v;
                self.tail = v;
            } else {
                // No elements in the queue we setup the initial state.
                self.head = v;
                self.tail = v;
            }
        }

        // Concatène une autre file `other` à la fin de celle-ci en O(1).
        pub fn pushAll(self: *Self, other: Self) void {
            if (self.tail) |tail| {
                tail.next = other.head;
            } else {
                self.head = other.head;
            }
            self.tail = other.tail;
        }

        // Retire et renvoie l'élément de tête, ou `null` si la file est vide.
        /// Dequeue the next element from the queue.
        pub fn pop(self: *Self) ?*T {
            // The next element is in "head".
            const next = self.head orelse return null;

            // If the head and tail are equal this is the last element
            // so we also set tail to null so we can now be empty.
            if (self.head == self.tail) self.tail = null;

            // Head is whatever is next (if we're the last element,
            // this will be null);
            self.head = next.next;

            // We set the "next" field to null so that this element
            // can be inserted again.
            next.next = null;
            return next;
        }

        // Renvoie le nombre d'éléments (parcours complet, O(n)).
        pub fn len(self: Self) usize {
            var ret: usize = 0;
            var current = self.head;
            while (current) |elem| : (current = elem.next) {
                ret += 1;
            }
            return ret;
        }

        // Renvoie vrai si la file est vide.
        /// Returns true if the queue is empty.
        pub fn empty(self: *const Self) bool {
            return self.head == null;
        }
    };
}

// Teste les opérations de base de la file SPSC.
test SPSC {
    const testing = std.testing;

    // Élément de test : maillon simple exposant un champ `next` vers le suivant.
    // Types
    const Elem = struct {
        const Self = @This();
        next: ?*Self = null,
    };
    const Queue = SPSC(Elem);
    var q: Queue = .{};
    try testing.expect(q.empty());

    // Elems
    var elems: [10]Elem = @splat(.{});

    // One
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    try testing.expect(!q.empty());
    try testing.expect(q.pop().? == &elems[0]);
    try testing.expect(q.pop() == null);
    try testing.expect(q.empty());

    // Two
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    q.push(&elems[1]);
    try testing.expect(q.pop().? == &elems[0]);
    try testing.expect(q.pop().? == &elems[1]);
    try testing.expect(q.pop() == null);

    // Interleaved
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    try testing.expect(q.pop().? == &elems[0]);
    q.push(&elems[1]);
    try testing.expect(q.pop().? == &elems[1]);
    try testing.expect(q.pop() == null);
}

// File d'attente intrusive MPSC (plusieurs producteurs, un consommateur), implémentation de type Vyukov.
/// An intrusive MPSC (multi-provider, single consumer) queue implementation.
/// The type T must have a field "next" of type `?*T`.
///
/// This is an implementatin of a Vyukov Queue[1].
/// TODO(mitchellh): I haven't audited yet if I got all the atomic operations
/// correct. I was short term more focused on getting something that seemed
/// to work; I need to make sure it actually works.
///
/// For those unaware, an intrusive variant of a data structure is one in which
/// the data type in the list has the pointer to the next element, rather
/// than a higher level "node" or "container" type. The primary benefit
/// of this (and the reason we implement this) is that it defers all memory
/// management to the caller: the data structure implementation doesn't need
/// to allocate "nodes" to contain each element. Instead, the caller provides
/// the element and how its allocated is up to them.
///
/// [1]: https://www.1024cores.net/home/lock-free-algorithms/queues/intrusive-mpsc-node-based-queue
pub fn MPSC(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Head is the front of the queue and tail is the back of the queue.
        head: *T,
        tail: *T,
        stub: T,

        // Initialise la file (pointeur stable requis) ; à appeler avant tout usage concurrent.
        /// Initialize the queue. This requires a stable pointer to itself.
        /// This must be called before the queue is used concurrently.
        pub fn init(self: *Self) void {
            self.head = &self.stub;
            self.tail = &self.stub;
            self.stub.next = null;
        }

        // Ajoute `v` en tête ; appelable par plusieurs producteurs.
        /// Push an item onto the queue. This can be called by any number
        /// of producers.
        pub fn push(self: *Self, v: *T) void {
            @atomicStore(?*T, &v.next, null, .unordered);
            const prev = @atomicRmw(*T, &self.head, .Xchg, v, .acq_rel);
            @atomicStore(?*T, &prev.next, v, .release);
        }

        // Retire le premier élément ; appelable par un seul consommateur à la fois.
        /// Pop the first in element from the queue. This must be called
        /// by only a single consumer at any given time.
        pub fn pop(self: *Self) ?*T {
            var tail = @atomicLoad(*T, &self.tail, .unordered);
            var next_ = @atomicLoad(?*T, &tail.next, .acquire);
            if (tail == &self.stub) {
                const next = next_ orelse return null;
                @atomicStore(*T, &self.tail, next, .unordered);
                tail = next;
                next_ = @atomicLoad(?*T, &tail.next, .acquire);
            }

            if (next_) |next| {
                @atomicStore(*T, &self.tail, next, .release);
                tail.next = null;
                return tail;
            }

            const head = @atomicLoad(*T, &self.head, .unordered);
            if (tail != head) return null;
            self.push(&self.stub);

            next_ = @atomicLoad(?*T, &tail.next, .acquire);
            if (next_) |next| {
                @atomicStore(*T, &self.tail, next, .unordered);
                tail.next = null;
                return tail;
            }

            return null;
        }
    };
}

// Teste les opérations de base de la file MPSC.
test MPSC {
    const testing = std.testing;

    // Élément de test : maillon simple exposant un champ `next` vers le suivant.
    // Types
    const Elem = struct {
        const Self = @This();
        next: ?*Self = null,
    };
    const Queue = MPSC(Elem);
    var q: Queue = undefined;
    q.init();

    // Elems
    var elems: [10]Elem = @splat(.{});

    // One
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    try testing.expect(q.pop().? == &elems[0]);
    try testing.expect(q.pop() == null);

    // Two
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    q.push(&elems[1]);
    try testing.expect(q.pop().? == &elems[0]);
    try testing.expect(q.pop().? == &elems[1]);
    try testing.expect(q.pop() == null);

    // // Interleaved
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    try testing.expect(q.pop().? == &elems[0]);
    q.push(&elems[1]);
    try testing.expect(q.pop().? == &elems[1]);
    try testing.expect(q.pop() == null);
}

// File circulaire de capacité fixe `size` stockée dans un tableau.
pub fn ArrayQueue(comptime T: type, comptime size: usize) type {
    return struct {
        const Self = @This();

        vals: [size]T = undefined,
        head: ?usize = null,
        tail: ?usize = null,

        // Renvoie le nombre d'éléments présents.
        pub fn len(self: Self) usize {
            switch (self.state()) {
                .empty => return 0,
                .one => return 1,
                .many => {
                    const head = self.head.?;
                    const tail = self.tail.?;
                    if (tail > head) {
                        return tail - head + 1;
                    }
                    return size - head + tail + 1;
                },
            }
        }

        // Renvoie l'espace libre restant.
        pub fn available(self: Self) usize {
            return size - self.len();
        }

        // Ajoute `val` ; erreur `QueueFull` si la file est pleine.
        pub fn push(self: *Self, val: T) !void {
            if (self.len() == size) {
                return error.QueueFull;
            }
            switch (self.state()) {
                .empty => {
                    self.head = 0;
                    self.tail = 0;
                    self.vals[0] = val;
                },
                .one, .many => {
                    const tail = self.tail.?;
                    const new_tail = (tail + 1) % size;
                    self.vals[new_tail] = val;
                    self.tail = new_tail;
                },
            }
        }

        // Retire et renvoie l'élément de tête, ou `null` si vide.
        pub fn pop(self: *Self) ?T {
            switch (self.state()) {
                .empty => return null,
                .one => {
                    const out = self.vals[self.head.?];
                    self.head = null;
                    self.tail = null;
                    return out;
                },
                .many => {
                    const out = self.vals[self.head.?];
                    self.head = (self.head.? + 1) % size;
                    return out;
                },
            }
        }

        // État interne de la file circulaire : vide, un élément, ou plusieurs.
        const State = enum { empty, one, many };
        // Détermine l'état interne (vide, un élément, plusieurs).
        inline fn state(self: Self) State {
            if (self.head == null) return .empty;
            if (self.head.? == self.tail.?) return .one;
            return .many;
        }
    };
}
