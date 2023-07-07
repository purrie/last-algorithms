const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expectString = std.testing.expectEqualStrings;
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const DebugAllocator = std.heap.GeneralPurposeAllocator(.{ .never_unmap = true, .retain_metadata = true });

pub fn main() !void {
    // Prints to stderr (it's a shortcut based on `std.io.getStdErr()`)
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});

    // stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try bw.flush(); // don't forget to flush!
}

fn linearSearch(comptime T : type, array : []const T, element : T) ?usize {
    for (array, 0..) |e, i| {
        if (e == element) {
            return i;
        }
    }
    return null;
}

test "linear search" {
    const elements = [_]u32 { 1, 2, 3, 4, 5, 6 };

    const index = linearSearch(u32, &elements, 3).?;
    const expected : usize = 2;

    try expectEqual(expected, index);
}

fn binarySearch(comptime T : type, array : []const T, element : T) ?usize {
    var high : usize = array.len;
    var low : usize = 0;
    while (high > low) {
        const middle = low + (high - low) / 2;
        const value = array[middle];

        if (value == element) {
            return middle;
        }
        else if (element > value) {
            low = middle + 1;
        }
        else {
            high = middle;
        }
    }
    return null;
}

test "binary search" {
    const elements = [_]u32 { 1, 2, 3, 4, 5, 6 };

    const index = binarySearch(u32, &elements, 3).?;
    const expected : usize = 2;

    try expectEqual(expected, index);
}

fn sqrtSearch (comptime T : type, array : []const T, element : T) ?usize {
    const stride = @intFromFloat(usize, @sqrt(@floatFromInt(f32, array.len)));

    var i : usize = stride;
    while (i < array.len) : (i += stride) {
        if (T == bool) {
            if (array[i] == element) {
                break;
            }
        }
        else {
            if (array[i] > element) {
                break;
            }
        }
    }
    i -= stride;
    for (array[i..][0..stride], i..) |item, index| {
        if (item == element) {
            return index;
        }
    }

    return null;
}

test "sqrt search" {
    const elements = [_]u32 { 1, 2, 3, 4, 5, 6 };
    const bools = [_]bool { false } ** 3 ++ [_]bool {true} ** 4;

    const index = sqrtSearch(u32, &elements, 3).?;
    const expected : usize = 2;

    const b_index = sqrtSearch(bool, &bools, true).?;
    const b_expected : usize = 3;

    try expectEqual(expected, index);
    try expectEqual(b_expected, b_index);
}

fn bubbleSort (comptime T : type, array : []T) void {
    var last = array.len - 1;

    while (last > 0) : (last -= 1) {
        for (0..last) |i| {
            if (array[i] > array[i + 1]) {
                var t : T = array[i];
                array[i] = array[i + 1];
                array[i + 1] = t;
            }
        }
    }
}

test "bubble sort" {
    var elements = [_]u32 { 6, 5, 4, 3, 2, 1 };
    const expecteds = [_]u32 { 1, 2, 3, 4, 5, 6 };

    bubbleSort(u32, &elements);

    for (elements, expecteds) |element, expected| {
        try expectEqual(expected, element);
    }
}

pub fn createQueue (comptime T : type) type {
    const Node = struct {
        value : T,
        next : ?*@This(),

        fn create (value : T, allocator : std.mem.Allocator) !*@This() {
            var self = try allocator.create(@This());
            self.next = null;
            self.value = value;
            return self;
        }
    };
    return struct {

        length : usize,
        head : ?*Node,
        tail : ?*Node,
        allocator : std.mem.Allocator,

        pub fn init (allocator : std.mem.Allocator) @This() {
            return .{
                .length = 0,
                .head = null,
                .tail = null,
                .allocator = allocator,
            };
        }

        pub fn deinit (self : *@This()) void {
            while (self.head) |head| {
                self.head = head.next;
                self.allocator.destroy(head);
            }
            self.tail = null;
        }

        pub fn enqueue (self : *@This(), value : T) !void {
            var next = try Node.create(value, self.allocator);
            self.length += 1;

            if (self.tail) |tail| {
                tail.next = next;
                self.tail = next;
            } else {
                self.tail = next;
                self.head = next;
            }
        }

        pub fn deque (self : *@This()) ?T {
            if (self.head) |h| {
                self.length -= 1;
                defer self.allocator.destroy(h);

                var r = h.value;
                self.head = h.next;
                if (h.next == null) {
                    self.tail = null;
                }
                return r;
            }
            return null;
        }

        pub fn peek (self : *const @This()) ?T {
            if (self.head) |h| {
                return h.value;
            }
            return null;
        }

    };
}

test "queue" {
    const Queue = createQueue(u32);
    var alloc = std.heap.page_allocator;
    var queue = Queue.init(alloc);
    defer queue.deinit();

    for (0..5) |i| {
        try queue.enqueue(@intCast(u32, i));
    }

    var i : u32 = 0;
    while (queue.deque()) |item| : (i += 1) {
        try expectEqual(i, item);
    }
}

pub fn createStack (comptime T : type) type {
    const Node = struct {
        value : T,
        prev : ?*@This(),

        pub fn create (value : T, allocator : std.mem.Allocator) !*@This() {
            var r = try allocator.create(@This());
            r.value = value;
            r.prev = null;
            return r;
        }
    };
    return struct {
        top : ?*Node,
        length : usize,
        allocator : std.mem.Allocator,

        pub fn init (allocator : std.mem.Allocator) @This() {
            return .{
                .top = null,
                .length = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit (self : *@This()) void {
            while (self.top) |top| {
                self.top = top.prev;
                self.allocator.destroy(top);
            }
        }

        pub fn push (self : *@This(), value : T) !void {
            var next = try Node.create(value, self.allocator);
            self.length += 1;
            if (self.top) |top| {
                next.prev = top;
                self.top = next;
            } else {
                self.top = next;
            }
        }

        pub fn pop (self : *@This()) ?T {
            if (self.top) |top| {
                defer self.allocator.destroy(top);

                self.length -= 1;
                self.top = top.prev;
                return top.value;
            }
            return null;
        }

        pub fn peek (self : *@This()) ?T {
            if (self.top) |top| {
                return top.value;
            }
            return null;
        }
    };
}


test "stack" {
    const Stack = createStack(u32);
    var alloc = std.heap.page_allocator;

    var stack = Stack.init(alloc);
    defer stack.deinit();

    for (0..5) |i| {
        try stack.push(@intCast(u32, i));
    }

    const expected = [_]u32 { 4, 3, 2, 1, 0 };

    for (expected) |value| {
        var pop = stack.pop().?;
        try expectEqual(value, pop);
    }
    try std.testing.expect(stack.pop() == null);
}

pub fn createRingList(comptime T : type) type {
    return struct {
        allocator : std.mem.Allocator,

        items : []T,
        low   : usize,
        high  : usize,

        pub fn init (capacity : usize, allocator : std.mem.Allocator) !@This() {
            if (capacity < 2)
                return error.InitialCapacityTooSmall;

            const half = capacity / 2;
            return .{
                .low = half,
                .high = half,
                .allocator = allocator,
                .items = try allocator.alloc(T, capacity),
            };
        }

        pub fn deinit (self : *@This()) void {
            self.allocator.free(self.items);
        }

        pub fn deinitRecursive (self : *@This()) void {
            switch (@typeInfo(T)) {
                .Struct => {
                    if (@hasDecl(T, "deinitRecursive")) {
                        var len = self.length();
                        while (len > 0) : (len -= 1) {
                            var item = self.pop().?;
                            item.deinitRecursive();
                        }
                    }
                    else if (@hasDecl(T, "deinit")) {
                        var len = self.length();
                        while (len > 0) : (len -= 1) {
                            var item = self.pop().?;
                            item.deinit();
                        }
                    }
                    else {
                        @compileError("To recursively deinitialize a Ring List, the contained object must have a public 'deinit' or 'deinitRecursive' function");
                    }
                },
                else => {}
            }
            self.deinit();
        }

        fn grow (self : *@This(), amount : usize) !void {
            if (amount == 0)
                return error.GrowByZero;
            const new_cap = self.items.len + amount;
            var new = try self.allocator.alloc(T, new_cap);

            const len = self.length();
            const start = ( new_cap - len ) / 2;
            if (self.low < self.high) {
                @memcpy(new[start..][0..len], self.items[self.low..self.high]);
            }
            else if (self.low > self.high) {
                const top_len = self.items.len - self.low;
                @memcpy(new[start..][0..top_len], self.items[self.low..]);
                @memcpy(new[start + top_len..][0..len - top_len], self.items[0..self.high]);
            }
            self.allocator.free(self.items);
            self.items = new;
            self.low = start;
            self.high = start + len;
        }

        const enqueue = push;
        const enqueueBack = push;
        const dequeBack = pop;
        const dequeFront = deque;

        pub fn get (self : *@This(), index : usize) ?T {
            const len = self.length();
            if (index < len) {
                const real_index = ( self.low + index ) % self.items.len;
                return self.items[real_index];
            }
            return null;
        }

        pub fn length (self : *const @This()) usize {
            if (self.low == self.high)
                return 0;
            if (self.low > self.high)
                return self.high + (self.items.len - self.low);
            return self.high - self.low;
        }

        pub fn enqueueFront (self : *@This(), value : T) !void {
            if (self.length() == self.items.len - 1) {
                try self.grow(6);
            }
            if (self.low == 0) {
                self.low = self.items.len - 1;
            }
            else {
                self.low -= 1;
            }
            self.items[self.low] = value;
        }

        pub fn deque (self : *@This()) ?T {
            if (self.length() > 0) {
                defer self.low = (self.low + 1) % self.items.len;
                return self.items[self.low];
            }
            return null;
        }

        pub fn push (self : *@This(), value : T) !void {
            if (self.length() == self.items.len - 1) {
                try self.grow(6);
            }
            self.items[self.high] = value;
            self.high = (self.high + 1) % self.items.len;
        }

        pub fn pop (self : *@This()) ?T {
            const len = self.length();
            if (len > 0) {
                if (self.high == 0) {
                    self.high = self.items.len - 1;
                }
                else {
                    self.high -= 1;
                }
                return self.items[self.high];
            }
            return null;
        }
    };
}

test "ring list" {
    const Ring = createRingList(u32);
    var ring = try Ring.init(6, std.heap.page_allocator);
    defer ring.deinit();

    const SRing = createRingList(Ring);
    var sring = try SRing.init(2, std.heap.page_allocator);
    defer sring.deinitRecursive();

    const queue = [_]u32 {1, 3, 4, 5, 8};
    for (queue) |item| {
        try ring.enqueue(item);
    }
    for (queue) |expected| {
        try expectEqual(expected, ring.deque().?);
    }
    try std.testing.expect(ring.deque() == null);

    const stack = [_]u32 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for (stack) |item| {
        try ring.push(item);
    }
    const expected_low : usize = 3;
    const expected_hig : usize = 1;
    try expectEqual(expected_low, ring.low);
    try expectEqual(expected_hig, ring.high);

    for (0..stack.len) |index| {
        const top = stack.len - index - 1;
        const pop = ring.pop().?;
        try expectEqual(stack[top], pop);
    }
    try std.testing.expect(ring.pop() == null);
}

const Point = struct {
    x : usize,
    y : usize,

    pub fn equals ( self : Point, other : Point ) bool {
        return self.x == other.x and self.y == other.y;
    }

    pub fn add ( self : Point, diff : Diff ) Point {
        var result = Point{
            .x = @intCast(usize, @intCast(isize, self.x) + diff.x),
            .y = @intCast(usize, @intCast(isize, self.y) + diff.y),
        };
        return result;
    }
};
const Diff = struct {
    x : isize,
    y : isize,
};

const Points = std.ArrayList(Point);
const directions : [4]Diff = .{
    .{ .x = -1,.y =  0 },
    .{ .x = 1, .y = 0 },
    .{ .x = 0, .y = -1 },
    .{ .x = 0, .y = 1 }
};

fn walk (maze : [][]const u8, wall : u8, current : Point, end : Point, seen : [][]bool, path : *Points) !bool {
    if (current.x < 0 or current.x >= maze[0].len or
        current.y < 0 or current.y >= maze.len) {
        return false;
    }

    if (maze[current.y][current.x] == wall) {
        return false;
    }

    if (current.equals(end)) {
        try path.append(current);
        return true;
    }

    if (seen[current.y][current.x]) {
        return false;
    }

    seen[current.y][current.x] = true;
    try path.append(current);

    for (directions) |dir| {
        const new_cur = current.add(dir);
        const t = try walk(maze, wall, new_cur, end, seen, path);
        if (t) {
            return true;
        }
    }

    _ = path.popOrNull();
    return false;
}

fn mazeSolver (maze : [][]const u8, wall : u8, start : Point, end : Point, alloc : Allocator) !Points {
    var seen : [][]bool = try alloc.alloc([]bool, maze.len);
    for (0..maze.len) |i| {
        seen[i] = try alloc.alloc(bool, maze[i].len);
        @memset(seen[i], false);
    }
    defer {
        for (seen) |row| {
            alloc.free(row);
        }
        alloc.free(seen);
    }

    var points : Points = Points.init(alloc);
    errdefer points.deinit();

    const t = try walk(maze, wall, start, end, seen, &points);
    if (t) {
        return points;
    }

    return error.EndNotFound;
}

test "maze solver" {
    // not sure how to make this as const without zig freaking out
    var maze = [_][]const u8{
        "#####E#",
        "#     #",
        "#S#####",
    };
    const start = Point{ .x = 1, .y = 2};
    const end   = Point{ .x = 5, .y = 0};
    const wall  = '#';
    var da = DebugAllocator{};
    defer _ = da.deinit();
    var alloc = da.allocator();

    var points = try mazeSolver(&maze, wall, start, end, alloc);
    const expected = [7]Point{
        Point{.x = 1, .y = 2},
        Point{.x = 1, .y = 1},
        Point{.x = 2, .y = 1},
        Point{.x = 3, .y = 1},
        Point{.x = 4, .y = 1},
        Point{.x = 5, .y = 1},
        Point{.x = 5, .y = 0},
    };

    errdefer points.deinit();

    for (expected, points.items) |exp, po| {
        try expectEqual(exp, po);
    }

    points.deinit();

    try expect(da.detectLeaks() == false);
}

fn quicksort ( comptime T : type, array : []T ) void {
    if (array.len < 2)
        return;

    const last = array.len - 1;
    const pivot = array[last];
    var index : usize = 0;

    for (0..last) |i| {
        if (array[i] <= pivot) {
            const temp = array[i];
            array[i] = array[index];
            array[index] = temp;
            index += 1;
        }
    }

    array[last] = array[index];
    array[index] = pivot;

    quicksort(T, array[0..index]);
    quicksort(T, array[index + 1..]);
}

test "quicksort" {
    var array = [_]u32{ 3, 8, 2, 9, 7, 5, 6, 1, 4, 10 };
    const expected = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    quicksort(u32, array[0..]);

    for (array, expected) |item, exp| {
        try expectEqual(exp, item);
    }
}


pub fn createDoublyLinkedList (comptime T : type) type {
    return struct {
        const Node = struct {
            value : T,
            next : ?*Node,
            prev : ?*Node,
        };
        const Self = @This();

        allocator : Allocator,
        length : usize,
        head : ?*Node,
        tail : ?*Node,

        pub fn init (allocator : Allocator) Self {
            return .{
                .allocator = allocator,
                .length = 0,
                .head = null,
                .tail = null,
            };
        }

        pub fn deinit (self : *Self) void {
            var node = self.head;
            while (node) |n| {
                node = n.next;
                self.allocator.destroy(n);
            }
        }

        pub fn length (self : *Self) usize {
            return self.length;
        }
        pub fn insertAt (self : *Self, item : T, index : usize) !void {
            if (index > self.length) {
                return error.IndexOutofBounds;
            } else if (index == self.length) {
                return self.append(item);
            } else if (index == 0) {
                return self.prepend(item);
            }
            var new = try self.allocator.create(Node);
            new.value = item;
            errdefer self.allocator.destroy(new);

            var node = self.getAt(index);
            if (node) |n| {
                self.length += 1;
                new.next = n;
                new.prev = n.prev;
                new.next.?.prev = new;
                new.prev.?.next = new;
            }
            else {
                return error.InsertionFailed;
            }
        }
        pub fn remove (self : *Self, item : T) ?T {
            var node = self.head;
            while (node) |nod| : (node = nod.next) {
                // Comparison here only works with primitive types.
                // Support for all types is omited for briefety
                if (nod.value == item) {
                    if (nod.prev != null and nod.next != null) {
                        nod.prev.?.next = nod.next;
                        nod.next.?.prev = nod.prev;
                    }
                    else if (nod.prev) |p| {
                        p.next = null;
                        self.tail = p;
                    }
                    else if (nod.next) |n| {
                        n.prev = null;
                        self.head = n;
                    }
                    else {
                        self.head = null;
                        self.tail = null;
                    }
                    self.allocator.destroy(nod);
                    self.length -= 1;
                    return item;
                }
            }
            return null;
        }
        pub fn removeAt (self : *Self, index : usize) ?T {
            if (self.getAt(index)) |node| {
                if (node.prev != null and node.next != null) {
                    node.prev.?.next = node.next;
                    node.next.?.prev = node.prev;
                }
                else if (node.prev) |p| {
                    p.next = null;
                    self.tail = p;
                }
                else if (node.next) |n| {
                    n.prev = null;
                    self.head = n;
                }
                else {
                    self.head = null;
                    self.tail = null;
                }
                defer self.allocator.destroy(node);
                self.length -= 1;
                return node.value;
            }
            return null;
        }
        pub fn append (self : *Self, item : T) !void {
            var new = try self.allocator.create(Node);
            new.value = item;
            new.next = null;

            self.length += 1;
            if (self.tail) |tail| {
                tail.next = new;
                new.prev = tail;
            }
            else {
                new.prev = null;
                self.head = new;
                self.tail = new;
            }
        }
        pub fn prepend (self : *Self, item : T) !void {
            var node = try self.allocator.create(Node);
            self.length += 1;
            node.value = item;
            node.next = self.head;
            node.prev = null;
            if (self.head) |head| {
                head.prev = node;
            }
            else {
                self.tail = node;
            }
            self.head = node;
        }
        pub fn get (self : *Self, index : usize) ?T {
            if (self.getAt(index)) |n| {
                return n.value;
            }
            return null;
        }

        fn getAt(self : *Self, index : usize) ?*Node {
            if (index >= self.length)
                return null;

            var node = self.head;
            for (0..index) |_| {
                node = node.?.next;
            }
            return node;
        }
    };
}

test "doubly linked list" {
    const DLL = createDoublyLinkedList(u32);
    var dba = DebugAllocator{};
    var list = DLL.init(dba.allocator());

    try list.append(3);
    try list.append(5);
    try list.prepend(2);
    try list.prepend(1);
    try list.insertAt(4, 3);

    for (1..6) |i| {
        try expectEqual(@intCast(u32, i), list.get(i - 1).?);
    }

    try expectEqual(@intCast(u32, 5), list.remove(5).?);
    try expectEqual(@intCast(u32, 3), list.removeAt(2).?);
    try expectEqual(@intCast(u32, 1), list.remove(1).?);

    try expectEqual(@intCast(u32, 2), list.get(0).?);
    try expectEqual(@intCast(u32, 4), list.get(1).?);

    list.deinit();

    try expect(dba.deinit() == .ok);
}

pub fn createBinaryTree(comptime T : type) type {
    return struct {
        const Node = struct {
            value : T,
            left  : ?*Node,
            right : ?*Node,

            fn create (value : T, allocator : Allocator) !*Node {
                var n = try allocator.create(Node);
                n.left = null;
                n.right = null;
                n.value = value;
                return n;
            }
            fn deinit (self : *Node, allocator : Allocator) void {
                if (self.left) |left| {
                    left.deinit(allocator);
                }
                if (self.right) |right| {
                    right.deinit(allocator);
                }
                allocator.destroy(self);
            }
            fn preOrder (self : *Node, result : []T) []T {
                result[0] = self.value;
                var out = result[1..];
                if (self.left) |left| {
                    out = left.preOrder(out);
                }
                if (self.right) |right| {
                    out = right.preOrder(out);
                }
                return out;
            }
            fn inOrder (self : *Node, result : []T) []T {
                var out = result[0..];
                if (self.left) |left| {
                    out = left.preOrder(out);
                }
                out[0] = self.value;
                out = out[1..];
                if (self.right) |right| {
                    out = right.preOrder(out);
                }
                return out;
            }
            fn postOrder (self : *Node, result : []T) []T {
                var out = result[0..];
                if (self.left) |left| {
                    out = left.preOrder(out);
                }
                if (self.right) |right| {
                    out = right.preOrder(out);
                }
                out[0] = self.value;
                return out[1..];
            }
            fn equals (a : ?*Node, b : ?*Node) bool {
                if (a == null and b == null)
                    return true;
                if (a == null or b == null)
                    return false;
                if (a.?.value != b.?.value)
                    return false;

                return Node.equals(a.?.left, b.?.left) and Node.equals(a.?.right, b.?.right);
            }
            fn find (self : *Node, value : T) ?*Node {
                if (self.value == value) {
                    return self;
                }
                if (self.value >= value) {
                    if (self.left) |left| {
                        return left.find(value);
                    }
                }
                else if (self.right) |right| {
                    return right.find(value);
                }
                return null;
            }
            fn insert (self : *Node, value : T, allocator : Allocator) !void {
                if (self.value >= value) {
                    if (self.left) |left| {
                        try left.insert(value, allocator);
                    }
                    else {
                        self.left = try Node.create(value, allocator);
                    }
                }
                else {
                    if (self.right) |right| {
                        try right.insert(value, allocator);
                    }
                    else {
                        self.right = try Node.create(value, allocator);
                    }
                }
            }
            fn delete (self : *Node, value : T, allocator : Allocator) ?*Node {
                if (self.value == value) {
                    if (self.left) |left| {
                        var large_parent = self;
                        var large = left;
                        while (large.right) |right| {
                            large_parent = large;
                            large = right;
                        }
                        self.value = large.value;

                        if (large.left) |lleft| {
                            large_parent.right = lleft;
                        }
                        else {
                            large_parent.right = null;
                        }

                        return large;
                    }
                    else if (self.right) |right| {
                        self.value = right.value;
                        self.right = right.right;
                        self.left  = right.left;

                        return right;
                    }
                    return self;
                }
                else if (self.value >= value) {
                    if (self.left) |left| {
                        if (left.delete(value, allocator)) |del| {
                            if (del == left)
                                self.left = null;
                            return del;
                        }
                    }
                }
                else {
                    if (self.right) |right| {
                        if (right.delete(value, allocator)) |del| {
                            if (del == right)
                                self.right = null;
                            return del;
                        }
                    }
                }
                return null;
            }
        };
        const Self = @This();

        allocator : Allocator,
        head : ?*Node,
        count : usize,

        pub fn init (allocator : Allocator) Self {
            return .{
                .allocator = allocator,
                .head = null,
                .count = 0,
            };
        }
        pub fn deinit (self : *Self) void {
            if (self.head) |head| {
                head.deinit(self.allocator);
                self.head = null;
            }
        }
        pub fn elements (self : *Self) usize {
            return self.count;
        }
        pub fn preOrder(self : *const Self, result : []T) !void {
            if (result.len < self.count) {
                return error.ResultArrayTooSmall;
            }
            if (self.head) |head| {
                _ = head.preOrder(result);
            }
            else {
                return error.TreeEmpty;
            }
        }
        pub fn inOrder(self : *const Self, result : []T) !void {
            if (result.len < self.count) {
                return error.ResultArrayTooSmall;
            }
            if (self.head) |head| {
                _ = head.inOrder(result);
            }
            else {
                return error.TreeEmpty;
            }
        }
        pub fn postOrder(self : *const Self, result : []T) !void {
            if (result.len < self.count) {
                return error.ResultArrayTooSmall;
            }
            if (self.head) |head| {
                _ = head.postOrder(result);
            }
            else {
                return error.TreeEmpty;
            }
        }
        pub fn breadthFirstSearchExists (self : *const Self, element : T) !bool {
            if (self.count == 0)
                return false;

            var queue = createQueue(*Node).init(self.allocator);
            defer queue.deinit();

            try queue.enqueue(self.head.?);

            while (queue.deque()) |item| {
                if (item.value == element)
                    return true;

                if (item.left) |left| {
                    try queue.enqueue(left);
                }
                if (item.right) |right| {
                    try queue.enqueue(right);
                }
            }

            return false;
        }
        pub fn compareTreeEquals (self : *const Self, other : *const Self) bool {
            return Node.equals(self.head, other.head);
        }
        pub fn find (self : *const Self, item : T) bool {
            if (self.head) |head| {
                return head.find(item) != null;
            }
            return false;
        }
        pub fn insert (self : *Self, item : T) !void {
            if (self.head) |head| {
                try head.insert(item, self.allocator);
            }
            else {
                self.head = try Node.create(item, self.allocator);
            }
            self.count += 1;
        }
        pub fn delete (self : *Self, item : T) void {
            if (self.head) |head| {
                if (head.delete(item, self.allocator)) |node| {
                    if (node == head) {
                        self.head = null;
                    }
                    self.allocator.destroy(node);
                    self.count -= 1;
                }
            }
        }
    };
}

test "binary tree traversal in pre order" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree = BTree.init(alloc);

    // . . .
    tree.head = try BTree.Node.create(1, alloc);
    tree.head.?.left = try BTree.Node.create(2, alloc);
    tree.head.?.right = try BTree.Node.create(3, alloc);
    tree.count = 3;

    var order = try alloc.alloc(u32, tree.elements());
    try tree.preOrder(order[0..]);

    for (order, 1..) |item, i| {
        try expectEqual(@intCast(u32, i), item);
    }

    alloc.free(order);
    tree.deinit();

    try expect(da.deinit() == .ok);
}

test "binary tree traversal in order" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree = BTree.init(alloc);

    // . . .
    tree.head = try BTree.Node.create(2, alloc);
    tree.head.?.left = try BTree.Node.create(1, alloc);
    tree.head.?.right = try BTree.Node.create(3, alloc);
    tree.count = 3;

    var order = try alloc.alloc(u32, tree.elements());
    try tree.inOrder(order[0..]);

    for (order, 1..) |item, i| {
        try expectEqual(@intCast(u32, i), item);
    }

    alloc.free(order);
    tree.deinit();

    try expect(da.deinit() == .ok);
}

test "binary tree traversal in post order" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree = BTree.init(alloc);

    // . . .
    tree.head = try BTree.Node.create(3, alloc);
    tree.head.?.left = try BTree.Node.create(1, alloc);
    tree.head.?.right = try BTree.Node.create(2, alloc);
    tree.count = 3;

    var order = try alloc.alloc(u32, tree.elements());
    try tree.postOrder(order[0..]);

    for (order, 1..) |item, i| {
        try expectEqual(@intCast(u32, i), item);
    }

    alloc.free(order);
    tree.deinit();

    try expect(da.deinit() == .ok);
}

test "binary tree breadth first search" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree = BTree.init(alloc);

    // . . .
    tree.head = try BTree.Node.create(1, alloc);
    tree.head.?.left = try BTree.Node.create(2, alloc);
    tree.head.?.right = try BTree.Node.create(3, alloc);
    tree.count = 3;

    try expect(try tree.breadthFirstSearchExists(2));

    tree.deinit();

    try expect(da.deinit() == .ok);

}
test "binary tree compare" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree1 = BTree.init(alloc);
    var tree2 = BTree.init(alloc);
    var tree3 = BTree.init(alloc);

    tree1.head = try BTree.Node.create(1, alloc);
    tree1.head.?.left = try BTree.Node.create(2, alloc);
    tree1.head.?.right = try BTree.Node.create(3, alloc);
    tree1.count = 3;

    tree2.head = try BTree.Node.create(1, alloc);
    tree2.head.?.left = try BTree.Node.create(2, alloc);
    tree2.head.?.right = try BTree.Node.create(3, alloc);
    tree2.count = 3;

    tree3.head = try BTree.Node.create(1, alloc);
    tree3.head.?.left = try BTree.Node.create(2, alloc);
    tree3.head.?.left.?.left = try BTree.Node.create(3, alloc);
    tree3.count = 3;

    try expect(tree1.compareTreeEquals(&tree2));
    try expect(tree1.compareTreeEquals(&tree3) == false);

    tree1.deinit();
    tree2.deinit();
    tree3.deinit();

    try expect(da.deinit() == .ok);

}

test "binary tree depth first find" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree = BTree.init(alloc);

    // . . .
    tree.head = try BTree.Node.create(2, alloc);
    tree.head.?.left = try BTree.Node.create(1, alloc);
    tree.head.?.right = try BTree.Node.create(3, alloc);
    tree.count = 3;

    try expect(tree.find(2));
    try expect(tree.find(4) == false);

    tree.deinit();

    try expect(da.deinit() == .ok);

}

test "binary tree insert" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree = BTree.init(alloc);

    // . . .
    try tree.insert(1);
    try tree.insert(2);
    try tree.insert(3);

    var order = try alloc.alloc(u32, tree.elements());
    try tree.inOrder(order[0..]);

    for (order, 1..) |item, i| {
        try expectEqual(@intCast(u32, i), item);
    }

    tree.deinit();
    alloc.free(order);

    try expect(da.deinit() == .ok);

}

test "binary tree deletion" {
    const BTree = createBinaryTree(u32);
    var da = DebugAllocator{};
    var alloc = da.allocator();
    var tree = BTree.init(alloc);

    // . . .
    try tree.insert(3);
    try tree.insert(1);
    try tree.insert(2);
    try tree.insert(4);
    try tree.insert(5);
    try expectEqual(@intCast(usize, 5), tree.count);

    var order = try alloc.alloc(u32, tree.elements());
    try tree.inOrder(order[0..]);

    for (order, 1..) |item, i| {
        try expectEqual(@intCast(u32, i), item);
    }

    tree.delete(5);
    tree.delete(3);
    tree.delete(1);
    try expectEqual(@intCast(usize, 2), tree.count);

    try tree.inOrder(order[0..2]);
    try expectEqual(@intCast(u32, 2), order[0]);
    try expectEqual(@intCast(u32, 4), order[1]);

    tree.deinit();
    alloc.free(order);

    try expect(da.deinit() == .ok);

}

pub fn createMinHeap (comptime T : type) type {
    return struct {
        const Self = @This();
        alloc : Allocator,
        items : []T,
        cap : usize,

        pub fn init (cap : usize, alloc : Allocator) !Self {
            var self = Self{
                .alloc = alloc,
                .items = try alloc.alloc(T, cap),
                .cap = cap,
            };
            self.items.len = 0;
            return self;
        }
        pub fn deinit (self : *Self) void {
            self.alloc.free(self.items.ptr[0..self.cap]);
        }

        fn isLarger (a : T, b : T) bool {
            return switch (@typeInfo(T)) {
                .Int => a > b,
                .Struct => a.compare(b) > 0,
                else => {
                    @compileError("Unsupported type");
                }
            };
        }
        fn parentIndex (index : usize) usize {
            return (index - 1) / 2;
        }
        fn leftChild (index : usize) usize {
            return index * 2 + 1;
        }
        fn rightChild (index : usize) usize {
            return index * 2 + 2;
        }
        fn heapifyUp (self : *Self, index : usize) void {
            if (index == 0)
                return;

            const parent_index = parentIndex(index);
            const parent_value = self.items[parent_index];
            const my_value = self.items[index];

            const is_smaller = isLarger(parent_value, my_value);

            if (is_smaller) {
                self.items[parent_index] = my_value;
                self.items[index] = parent_value;
                self.heapifyUp(parent_index);
            }
        }
        fn heapifyDown (self : *Self, index : usize) void {
            if (self.items.len <= index) {
                return;
            }

            const p_value = self.items[index];
            const l_child = leftChild(index);
            const r_child = rightChild(index);

            const idx = switch (l_child < self.items.len) {
                true => switch (r_child < self.items.len) {
                    true => both: {
                        const r_v = self.items[r_child];
                        const l_v = self.items[l_child];
                        if (isLarger(l_v, r_v)) {
                            break :both .{r_v, r_child};
                        }
                        else {
                            break :both .{l_v, l_child};
                        }
                    },
                    false => .{self.items[l_child], l_child},
                },
                false => return,
            };

            const smaller = switch (@typeInfo(T)) {
                .Int => idx[0] < p_value,
                .Struct => idx[0].compare(p_value) < 0,
                else => {
                    @compileError("Unsupported type");
                }
            };

            if (smaller) {
                self.items[index] = idx[0];
                self.items[idx[1]] = p_value;
                self.heapifyDown(idx[1]);
            }
        }

        pub fn insert (self : *Self, value : T) !void {
            if (self.items.len == self.cap) {
                var new = try self.alloc.alloc(T, self.cap + 6);
                new.len = self.items.len;
                @memcpy(new, self.items);
                self.cap += 6;
                self.alloc.free(self.items);
                self.items = new;
            }
            const index = self.items.len;
            self.items.len += 1;
            self.items[index] = value;
            self.heapifyUp(index);
        }
        fn indexOf (self : *Self, value : T) ?usize {
            for (0..self.items.len) |i| {
                var found = switch (@typeInfo(T)) {
                    .Struct => self.items[i].equals(value),
                    .Int => self.items[i] == value,
                    else => @compileError("Index operation not supported for the type"),
                };
                if (found) {
                    return i;
                }
            }
            return null;
        }
        pub fn get (self : *Self, value : T) ?T {
            // TODO create a map with indexes
            if (self.indexOf(value)) |i| {
                return self.items[i];
            }
            return null;
        }
        pub fn update (self : *Self, value : T) void {
            if (self.indexOf(value)) |i| {
                if (isLarger(value, self.items[i])) {
                    self.items[i] = value;
                    self.heapifyDown(i);
                }
                else {
                    self.items[i] = value;
                    self.heapifyUp(i);
                }
            }
        }
        pub fn pop(self : *Self) ?T {
            if (self.items.len > 0) {
                const ret = self.items[0];
                const last = self.items.len - 1;
                self.items[0] = self.items[last];
                self.items.len -= 1;
                self.heapifyDown(0);
                return ret;
            }
            return null;
        }
    };
}

test "heap" {
    const Heap = createMinHeap(u32);
    var da = DebugAllocator{};
    defer _ = da.deinit();
    var heap = try Heap.init(1, da.allocator());
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(4);
    try heap.insert(3);
    try heap.insert(2);
    try heap.insert(1);
    try heap.insert(0);

    for (0..6) |i| {
        const p = heap.pop();
        try expectEqual(@intCast(u32, i), p.?);
    }
    try expect(null == heap.pop());
}

pub fn createTrie(comptime T : type) type {
    return struct {
        const Self = @This();
        const Word = std.ArrayList(T);
        const Words = std.ArrayList(Word);
        const Node = struct {
            is_word : bool,
            parent : ?*Node,
            children : []?*Node,
            depth : usize,

            fn indexOf (item : T) !usize {
                if (T == u8) {
                    if (item < 'a' or item > maxValue()) {
                        std.debug.print("{c} is out of bounds\n", .{item});
                        return error.ValueOutOfBounds;
                    }
                    return item - 'a';
                }
                else {
                    @compileError("Only u8 strings are supported here, for now");
                }
            }
            fn valueOf (index : usize) T {
                if (T == u8) {
                    return 'a' + @intCast(u8, index);
                }
                else {
                    @compileError("Only u8 strings are supported here, for now");
                }
            }
            fn maxValue () T {
                if (T == u8) {
                    return 'z';
                }
                else {
                    @compileError("Only u8 strings are supported here, for now");
                }
            }
            fn maxLength () usize {
                if (T == u8) {
                    return 'z' - 'a' + 1;
                }
                else {
                    @compileError("Only u8 strings are supported here, for now");
                }
            }
            fn init (parent : ?*Node, allocator : Allocator) !*Node {
                var n = try allocator.create(Node);
                n.is_word = false;
                n.parent = parent;
                n.children = try allocator.alloc(?*Node, maxLength());
                @memset(n.children, null);
                if (parent) |p| {
                    n.depth = p.depth + 1;
                }
                else {
                    n.depth = 0;
                }
                return n;
            }
            fn deinit (self : *Node, allocator : Allocator) void {
                for (self.children) |child| {
                    if (child) |c| {
                        c.deinit(allocator);
                    }
                }
                allocator.free(self.children);
                allocator.destroy(self);
            }
            fn purge (self : *Node, allocator : Allocator) bool {
                if (self.is_word)
                    return false;

                var purged = true;
                for (0..self.children.len) |i| {
                    if (self.children[i]) |child| {
                        if (child.purge(allocator)) {
                            child.deinit(allocator);
                            self.children[i] = null;
                        }
                        else {
                            purged = false;
                        }
                    }
                }
                return purged;
            }
            fn unword (self : *Node, allocator : Allocator) void {
                self.is_word = false;
                if (self.parent) |p| {
                    _ = p.purge(allocator);
                }
            }
            fn markWord (self : *Node) void {
                self.is_word = true;
            }
            fn get (self : *Node, allocator : Allocator, value : T) !*Node {
                const index = try indexOf(value);
                if (self.children[index] == null) {
                    self.children[index] = try Node.init(self, allocator);
                }
                return self.children[index].?;
            }
            fn has (self : *Node, value : T) !bool {
                return self.children[try indexOf(value)] != null;
            }
            fn getWords (self : *Node, prefix : []const T, char_stack : []usize, result : *Words) !void {
                if (self.is_word) {
                    var result_word = try result.addOne();
                    result_word.* = Word.init(result.allocator);
                    try result_word.ensureTotalCapacity(char_stack.len);
                    result_word.items.len = char_stack.len;
                    for (0..char_stack.len) |i| {
                        result_word.items[i] = valueOf(char_stack[i]);
                    }
                }
                const last = char_stack.len;
                var stack = char_stack;
                stack.len += 1;

                if (prefix.len > 0) {
                    const car = prefix[0];
                    const cdr = prefix[1..];
                    const idx = try indexOf(car);
                    if (self.children[idx]) |child| {
                        stack[last] = idx;
                        try child.getWords(cdr, stack, result);
                    }
                }
                else {
                    for (self.children, 0..) |child, index| {
                        if (child) |c| {
                            stack[last] = index;
                            try c.getWords(prefix, stack, result);
                        }
                    }
                }
            }
        };

        allocator : Allocator,
        max_depth : usize,
        root : *Node,

        pub fn init (allocator : Allocator) !Self {
            return .{
                .root = try Node.init(null, allocator),
                .allocator = allocator,
                .max_depth = 0,
            };
        }
        pub fn deinit (self : *Self) void {
            self.root.deinit(self.allocator);
        }
        pub fn maxDepth (self : *Self) usize {
            return self.max_depth;
        }
        pub fn addWord (self : *Self, word : []const T) !void {
            var node = self.root;
            for (word) |c| {
                node = try node.get(self.allocator, c);
            }
            node.markWord();
            if (node.depth > self.max_depth) {
                self.max_depth = node.depth;
            }
        }
        pub fn getWords (self : *Self, prefix : []const T, result : *Words) !void {
            var stack = try self.allocator.alloc(usize, self.max_depth);
            defer self.allocator.free(stack);

            try self.root.getWords(prefix, stack[0..0], result);
        }
        pub fn removeWord (self : *Self, word : []const T) bool {
            var node = self.root;
            for (word) |char| {
                var has = node.has(char) catch false;
                if (has) {
                    node = node.get(self.allocator, char) catch unreachable;
                }
                else {
                    return false;
                }
            }
            _ = node.unword(self.allocator);
            return true;
        }

    };
}

test "trie" {
    const Trie = createTrie(u8);
    var da = DebugAllocator{};
    var memalloc = da.allocator();
    defer _ = da.deinit();

    var trie = try Trie.init(memalloc);
    defer trie.deinit();

    var words = Trie.Words.init(memalloc);
    defer {
        for (words.items) |item| {
            item.deinit();
        }
        words.deinit();
    }

    try trie.addWord("cat");
    try trie.addWord("car");
    try trie.addWord("cattle");
    try trie.addWord("cabbage");
    try trie.addWord("zig");

    try trie.getWords("ca", &words);
    try expectString("cabbage", words.items[0].items);
    try expectString("car", words.items[1].items);
    try expectString("cat", words.items[2].items);
    try expectString("cattle", words.items[3].items);
    try expectEqual(@as(usize, 4), words.items.len);

    for (words.items) |item| {
        item.deinit();
    }
    words.items.len = 0;

    try expect(trie.removeWord("cabbage"));
    try expect(trie.removeWord("car"));

    try trie.getWords("ca", &words);
    try expectString("cat", words.items[0].items);
    try expectString("cattle", words.items[1].items);
    try expectEqual(@as(usize, 2), words.items.len);

    for (words.items) |item| {
        item.deinit();
    }
    words.items.len = 0;

    try trie.getWords("z", &words);
    try expectString("zig", words.items[0].items);
    try expectEqual(@as(usize, 1), words.items.len);
}

const UsizeList = std.ArrayList(usize);
const BoolArray = std.ArrayList(bool);
const MaybeUsizeArray = std.ArrayList(?usize);

fn createMatrix (comptime size : usize, comptime T : type, comptime value : T) [size][size]T {
    var result : [size][size]T =  .{
        [_]T{value} ** size
    } ** size;
    return result;
}

fn breadthFirstSearchAdjacencyMatrix (comptime size : usize, comptime T : type, matrix : *const [size][size]T, source : usize, find : usize, allocator : Allocator) !?UsizeList {
    const UsizeQueue = createQueue(usize);

    var seen = BoolArray.init(allocator);
    var prev = MaybeUsizeArray.init(allocator);
    var queue = UsizeQueue.init(allocator);
    defer {
        seen.deinit();
        prev.deinit();
        queue.deinit();
    }

    try seen.appendNTimes(false, size);
    try prev.appendNTimes(null, size);
    seen.items[source] = true;

    try queue.enqueue(source);

    while (queue.deque()) |index| {
        if (index == find)
            break;

        var adjs = matrix[index];
        for (0..adjs.len) |connection| {
            if (adjs[connection] == 0) {
                continue;
            }
            if (seen.items[connection]) {
                continue;
            }

            seen.items[connection] = true;
            prev.items[connection] = index;
            try queue.enqueue(connection);
        }
    }

    if (prev.items[find] == null) {
        return null;
    }
    var result = UsizeList.init(allocator);
    errdefer result.deinit();

    var track_back = find;
    while (prev.items[track_back]) |idx| {
        try result.append(track_back);
        track_back = idx;
    }
    try result.append(source);
    const half = result.items.len / 2;
    for (0..half) |s| {
        const e = result.items.len - s - 1;
        const swap = result.items[s];
        result.items[s] = result.items[e];
        result.items[e] = swap;
    }
    return result;
}

test "breadth first search on adjacency matrix" {
    var da = DebugAllocator{};
    var alloc = da.allocator();
    defer _ = da.deinit();
    const MatrixType = u8;
    const matrix_size = 5;

    var matrix = createMatrix(matrix_size, MatrixType, 0);
    matrix[0][1] = 1;
    matrix[0][2] = 4;
    matrix[0][3] = 5;
    matrix[1][0] = 1;
    matrix[2][3] = 2;
    matrix[3][4] = 5;

    const start = 0;
    const end = 4;
    var path = try breadthFirstSearchAdjacencyMatrix(matrix_size, MatrixType, &matrix, start, end, alloc);
    defer path.?.deinit();

    const expected = [_]usize {0, 3, 4};
    for (path.?.items, expected) |p, e| {
        try expectEqual(e, p);
    }
}

const AdjacencyNode = struct {
    to : usize,
    weight : usize,
};
const AdjacencyConnections = std.ArrayList(AdjacencyNode);

const AdjacencyList = std.ArrayList(AdjacencyConnections);

fn dfsOnAlWalk (
    list : *const AdjacencyList,
    current : usize, end : usize,
    seen : *BoolArray, path : *UsizeList
) !bool {
    if (seen.items[current]) {
        return false;
    }

    try path.append(current);
    if (current == end) {
        return true;
    }

    seen.items[current] = true;

    for (list.items[current].items) |item| {
        if (try dfsOnAlWalk(list, item.to, end, seen, path)) {
            return true;
        }
    }
    path.items.len -= 1;
    return false;
}
fn depthFirstSearchAdjacencyList (
    list : *const AdjacencyList,
    start : usize, end : usize,
    allocator : Allocator
) !?UsizeList {
    var seen = BoolArray.init(allocator);
    defer seen.deinit();
    try seen.appendNTimes(false, list.items.len);
    var path = UsizeList.init(allocator);
    errdefer path.deinit();

    if (try dfsOnAlWalk(list, start, end, &seen, &path)) {
        return path;
    }
    path.deinit();
    return null;
}

test "depth first search on adjacency list" {
    var da = DebugAllocator{};
    var alloc = da.allocator();
    defer _ = da.deinit();

    var list = AdjacencyList.init(alloc);
    var node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 1, .weight = 1 });
    try node.append(.{ .to = 2, .weight = 4 });
    try node.append(.{ .to = 3, .weight = 5 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 0, .weight = 1 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 3, .weight = 2 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 4, .weight = 5 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    defer {
        for (list.items) |item| {
            item.deinit();
        }
        list.deinit();
    }

    const start : usize = 0;
    const end   : usize = 4;
    var path = try depthFirstSearchAdjacencyList(&list, start, end, alloc);
    try expect(path != null);
    defer path.?.deinit();

    const expected = [_]usize {0, 2, 3, 4};
    for (0..expected.len) |i| {
        try expectEqual(expected[i], path.?.items[i]);
    }
}

fn getLowestUnvisited (seen : *BoolArray, dists : *MaybeUsizeArray) ?usize {
    var index : ?usize = null;
    var lowest : usize = undefined;

    for (seen.items, dists.items, 0..) |s, d, i| {
        if (s) {
            continue;
        }
        if (d) |dist| {
            if (index) |_| {
                if (dist < lowest) {
                    index = i;
                    lowest = dist;
                }
            }
            else {
                index = i;
                lowest = dist;
            }
        }
    }
    return index;
}

fn dijkstraShortestPath (list : *const AdjacencyList, start : usize, end : usize, allocator : Allocator) !?UsizeList {
    var seen = BoolArray.init(allocator);
    try seen.appendNTimes(false, list.items.len);
    defer seen.deinit();

    var prev = MaybeUsizeArray.init(allocator);
    try prev.appendNTimes(null, list.items.len);
    defer prev.deinit();

    var dist = MaybeUsizeArray.init(allocator);
    try dist.appendNTimes(null, list.items.len);
    defer dist.deinit();
    dist.items[start] = 0;

    while (getLowestUnvisited(&seen, &dist)) |index| {
        seen.items[index] = true;

        const current = &list.items[index];
        for (current.items) |edge| {
            if (seen.items[edge.to]) {
                continue;
            }

            const distance = dist.items[index].? + edge.weight;
            if (dist.items[edge.to]) |old| {
                if (old > distance) {
                    dist.items[edge.to] = distance;
                    prev.items[edge.to] = index;
                }
            }
            else {
                dist.items[edge.to] = distance;
                prev.items[edge.to] = index;
            }
        }
    }

    if (prev.items[end] == null) {
        return null;
    }

    var path = UsizeList.init(allocator);
    errdefer path.deinit();
    var current = end;

    while (prev.items[current]) |i| {
        try path.append(current);
        current = i;
    }
    try path.append(start);

    const half = path.items.len / 2;
    for (0..half) |i| {
        const e = path.items.len - i - 1;
        const temp = path.items[i];
        path.items[i] = path.items[e];
        path.items[e] = temp;
    }

    return path;
}

fn dijkstraBetterPath (list : *const AdjacencyList, start : usize, end : usize, allocator : Allocator) !?UsizeList {
    const Node = struct {
        index : usize,
        weight : usize = std.math.maxInt(usize),

        pub fn compare (self : @This(), other : @This()) i32 {
            if (self.weight > other.weight) {
                return 1;
            }
            else if (self.weight < other.weight) {
                return -1;
            }
            return 0;
        }
        pub fn equals (self : @This(), other : @This()) bool {
            return self.index == other.index;
        }
    };
    var min = try createMinHeap(Node).init(list.items.len, allocator);
    defer min.deinit();

    for (0..list.items.len) |i| {
        if (i == start) {
            try min.insert(.{ .index = i, .weight = 0});
        }
        else {
            try min.insert(.{ .index = i });
        }
    }

    var prev = MaybeUsizeArray.init(allocator);
    try prev.appendNTimes(null, list.items.len);
    defer prev.deinit();

    while (min.pop()) |node| {
        for (list.items[node.index].items) |edge| {
            if (min.get(.{ .index = edge.to })) |lead| {
                var weight = node.weight + edge.weight;
                if (lead.weight > weight) {
                    prev.items[edge.to] = node.index;
                    min.update(.{ .index = lead.index, .weight = weight });
                }
            }
        }
    }
    if (prev.items[end] == null) {
        return null;
    }

    var path = UsizeList.init(allocator);
    errdefer path.deinit();

    var current = end;
    while (prev.items[current]) |from| {
        try path.append(current);
        current = from;
    }
    try path.append(start);

    const half = path.items.len / 2;
    for (0..half) |i| {
        const e = path.items.len - i - 1;
        const temp = path.items[i];
        path.items[i] = path.items[e];
        path.items[e] = temp;
    }

    return path;

}

test "dijkstra's shortest path" {
    var da = DebugAllocator{};
    var alloc = da.allocator();
    defer _ = da.deinit();

    var list = AdjacencyList.init(alloc);
    var node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 1, .weight = 1 });
    try node.append(.{ .to = 2, .weight = 4 });
    try node.append(.{ .to = 3, .weight = 5 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 0, .weight = 1 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 3, .weight = 2 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    try node.append(.{ .to = 4, .weight = 5 });
    node = try list.addOne();
    node.* = AdjacencyConnections.init(alloc);
    defer {
        for (list.items) |item| {
            item.deinit();
        }
        list.deinit();
    }

    const start : usize = 0;
    const end   : usize = 4;
    var path = try dijkstraBetterPath(&list, start, end, alloc);
    try expect(path != null);
    defer path.?.deinit();

    try expectEqual(@as(usize, 3), path.?.items.len);

    const expected = [_]usize {0, 3, 4};
    for (0..expected.len) |i| {
        try expectEqual(expected[i], path.?.items[i]);
    }
}

pub fn createLRU(comptime K : type, comptime V : type, comptime Context : type) type {
    return struct {
        const Self = @This();
        const Node = struct {
            value : V,
            next : ?*Node = null,
            prev : ?*Node = null,

            fn create (value : V, mem : Allocator) !*@This() {
                var self = try mem.create(@This());
                self.* = .{
                    .value = value
                };
                return self;
            }
        };
        const NodeContext = struct {
            pub fn hash (self : NodeContext, node : *Node) u32 {
                _ = self;
                var hash_v : u32 = @truncate(u32, @intFromPtr(node));
                return std.hash.murmur.Murmur2_32.hashUint32(hash_v);
            }
            pub fn eql (self : NodeContext, k1 : *Node, k2 : *Node, b_index : usize) bool {
                _ = b_index;
                _ = self;
                return k1 == k2;
            }
        };
        const Map = std.ArrayHashMapUnmanaged(K, *Node, Context, true);
        const ReverseMap = std.ArrayHashMapUnmanaged(*Node, K, NodeContext, true);

        allocator : Allocator,
        head : ?*Node,
        tail : ?*Node,
        length : usize,
        capacity : usize,

        lookup : Map,
        reverse_lookup : ReverseMap,

        pub fn init (cap : usize, alloc : Allocator) !Self {
            var lu = Map{};
            try lu.ensureTotalCapacity(alloc, cap);
            errdefer lu.deinit(alloc);
            var rl = ReverseMap{};
            try rl.ensureTotalCapacity(alloc, cap);
            return .{
                .capacity = cap,
                .head = null,
                .tail = null,
                .length = 0,
                .allocator = alloc,
                .lookup = lu,
                .reverse_lookup = rl,
            };
        }
        pub fn deinit (self : *Self) void {
            self.lookup.deinit(self.allocator);
            self.reverse_lookup.deinit(self.allocator);
            var curr = self.head;
            while (curr) |c| {
                curr = c.next;
                self.allocator.destroy(c);
            }
            self.head = null;
            self.tail = null;
            self.length = 0;
        }

        fn detach (self : *Self, node : *Node) void {
            if (self.head == node) {
                self.head = node.next;
            }
            if (self.tail == node) {
                self.tail = node.prev;
            }

            if (node.next) |next| {
                next.prev = node.prev;
                node.next = null;
            }
            if (node.prev) |prev| {
                prev.next = node.next;
                node.prev = null;
            }
        }
        fn prepend (self : *Self, node : *Node) void {
            if (self.head) |head| {
                node.next = head;
                head.prev = node;
                self.head = node;
            }
            else {
                self.head = node;
                self.tail = node;
            }
        }
        fn trimCache (self : *Self) void {
            if (self.length <= self.capacity) {
                return;
            }
            var tail = self.tail.?;
            self.detach(tail);
            const key = self.reverse_lookup.get(tail).?;
            _ = self.reverse_lookup.swapRemove(tail);
            _ = self.lookup.swapRemove(key);

            self.allocator.destroy(tail);
            self.length -= 1;
        }

        pub fn update (self : *Self, key : K, value : V) !void {
            const v = self.lookup.get(key);
            if (v) |val| {
                self.detach(val);
                self.prepend(val);
                val.value = value;
            }
            else {
                var node = try Node.create(value, self.allocator);
                self.prepend(node);
                self.length += 1;
                self.trimCache();

                try self.lookup.put(self.allocator, key, node);
                try self.reverse_lookup.put(self.allocator, node, key);
            }
        }
        pub fn get (self : *Self, key : K) ?V {
            const v = self.lookup.get(key);
            if (v) |val| {
                self.detach(val);
                self.prepend(val);
                return val.value;
            }
            return null;
        }
    };
}

test "least recent used cache" {
    const Context = std.array_hash_map.StringContext;
    const LRU = createLRU([]const u8, u32, Context);
    var da = DebugAllocator{};
    var mem = da.allocator();
    defer _ = da.deinit();

    var lru = try LRU.init(3, mem);
    defer lru.deinit();

    try lru.update("hi", 2);
    try lru.update("cat", 3);

    try expectEqual(@as(u32, 2), lru.get("hi").?);
    try expectEqual(@as(u32, 3), lru.get("cat").?);

    try lru.update("cattle", 4);
    try lru.update("cat", 6);

    try expectEqual(@as(u32, 4), lru.get("cattle").?);
    try expectEqual(@as(u32, 6), lru.get("cat").?);
    try expectEqual(@as(u32, 2), lru.get("hi").?);

    try lru.update("zig", 9);

    try expectEqual(@as(?u32, null), lru.get("cattle"));
    try expectEqual(@as(u32, 2), lru.get("hi").?);
    try expectEqual(@as(u32, 6), lru.get("cat").?);
    try expectEqual(@as(u32, 9), lru.get("zig").?);

}
