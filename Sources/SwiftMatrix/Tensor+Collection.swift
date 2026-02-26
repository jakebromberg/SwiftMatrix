extension Tensor: RandomAccessCollection {
    public var startIndex: Int { 0 }
    public var endIndex: Int { shape.reduce(1, *) }

    public subscript(linearIndex linearIndex: Int) -> Element {
        get { storage[storageIndex(forLinearIndex: linearIndex)] }
        set { storage[storageIndex(forLinearIndex: linearIndex)] = newValue }
    }

    public subscript(position: Int) -> Element {
        get { storage[storageIndex(forLinearIndex: position)] }
        set { storage[storageIndex(forLinearIndex: position)] = newValue }
    }

    public func index(after i: Int) -> Int { i + 1 }
    public func index(before i: Int) -> Int { i - 1 }
}

extension Tensor: Equatable where Element: Equatable {
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        lhs.shape == rhs.shape && Array(lhs) == Array(rhs)
    }
}

extension Tensor: Hashable where Element: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(shape)
        for element in self {
            hasher.combine(element)
        }
    }
}
