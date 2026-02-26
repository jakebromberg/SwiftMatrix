public struct Tensor<Element> {
    var storage: [Element]
    public let shape: [Int]
    public let strides: [Int]
    public let offset: Int

    public var rank: Int { shape.count }

    public init(shape: [Int], repeating repeatedValue: Element) {
        let count = shape.reduce(1, *)
        self.storage = Array(repeating: repeatedValue, count: count)
        self.shape = shape
        self.strides = Self.computeStrides(for: shape)
        self.offset = 0
    }

    public init(shape: [Int], elements: [Element]) {
        let count = shape.reduce(1, *)
        precondition(elements.count == count,
                     "Element count \(elements.count) does not match shape \(shape) (expected \(count))")
        self.storage = elements
        self.shape = shape
        self.strides = Self.computeStrides(for: shape)
        self.offset = 0
    }

    public init(_ arrays: [[Element]]) {
        let depth = arrays.count
        guard depth > 0 else {
            self.init(shape: [0, 0], elements: [])
            return
        }
        let width = arrays[0].count
        precondition(arrays.allSatisfy { $0.count == width },
                     "All rows must have the same length")
        self.init(shape: [depth, width], elements: arrays.flatMap { $0 })
    }

    init(storage: [Element], shape: [Int], strides: [Int], offset: Int) {
        self.storage = storage
        self.shape = shape
        self.strides = strides
        self.offset = offset
    }

    static func computeStrides(for shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var strides = Array(repeating: 1, count: shape.count)
        for i in Swift.stride(from: shape.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        return strides
    }

    var isContiguous: Bool {
        strides == Self.computeStrides(for: shape) && offset == 0
    }

    func storageIndex(for indices: [Int]) -> Int {
        precondition(indices.count == rank,
                     "Expected \(rank) indices, got \(indices.count)")
        return offset + zip(indices, strides).reduce(0) { $0 + $1.0 * $1.1 }
    }

    func storageIndex(forLinearIndex linear: Int) -> Int {
        guard !isContiguous else { return linear }
        var remaining = linear
        var storageIdx = offset
        for axis in 0..<rank {
            let logicalStride = shape[(axis + 1)...].reduce(1, *)
            let coord = remaining / logicalStride
            remaining %= logicalStride
            storageIdx += coord * strides[axis]
        }
        return storageIdx
    }
}

extension Tensor: Sendable where Element: Sendable {}
