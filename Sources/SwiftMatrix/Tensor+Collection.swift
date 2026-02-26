/// `RandomAccessCollection` conformance using a flat linear index.
///
/// The collection index is `Int`, representing a position in row-major iteration order.
/// This avoids the partial-order problem that ``Pair`` has with `Comparable` and enables
/// efficient random access.
extension Tensor: RandomAccessCollection {
    public var startIndex: Int { 0 }
    public var endIndex: Int { count }

    /// Accesses the element at the given linear index using an explicit argument label.
    ///
    /// This subscript disambiguates from the multi-dimensional ``subscript(_:)-6gkqj``
    /// when you want to explicitly use the flat linear index.
    ///
    /// - Parameter linearIndex: A position in `0..<count`, following row-major order.
    public subscript(linearIndex linearIndex: Int) -> Element {
        get { storage[storageIndex(forLinearIndex: linearIndex)] }
        set { storage[storageIndex(forLinearIndex: linearIndex)] = newValue }
    }

    /// Accesses the element at the given linear index (`Collection` conformance).
    ///
    /// For a tensor with shape `[2, 3]` and elements `[1, 2, 3, 4, 5, 6]`:
    /// - Position 0 is element `1` (at coordinates `[0, 0]`)
    /// - Position 3 is element `4` (at coordinates `[1, 0]`)
    ///
    /// - Parameter position: A valid index in `startIndex..<endIndex`.
    public subscript(position: Int) -> Element {
        get { storage[storageIndex(forLinearIndex: position)] }
        set { storage[storageIndex(forLinearIndex: position)] = newValue }
    }

    public func index(after i: Int) -> Int { i + 1 }
    public func index(before i: Int) -> Int { i - 1 }
}

/// Tensors are equal when they have the same shape and the same elements in the same positions.
///
/// Two tensors with different internal layouts (e.g., different strides from a transpose) but
/// the same logical shape and elements are considered equal.
extension Tensor: Equatable where Element: Equatable {
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        lhs.shape == rhs.shape && lhs.elementsEqual(rhs)
    }
}

/// Hashing incorporates both shape and elements, consistent with ``Equatable``.
extension Tensor: Hashable where Element: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(shape)
        for element in self {
            hasher.combine(element)
        }
    }
}
