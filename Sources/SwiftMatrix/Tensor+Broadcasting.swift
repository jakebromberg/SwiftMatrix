/// Broadcasting support for element-wise operations on tensors with compatible shapes.
///
/// Broadcasting follows NumPy semantics: shapes are aligned right-to-left, each dimension
/// must be equal or one of them must be 1. Dimensions of size 1 are "stretched" to match
/// the other shape by setting their stride to 0.

extension Tensor {

    /// Computes the broadcast-compatible output shape, or `nil` if incompatible.
    ///
    /// Shapes are aligned right-to-left and padded with 1s on the left for the shorter shape.
    /// Each pair of dimensions is compatible if they are equal or one is 1. The output
    /// dimension is the maximum of the two.
    ///
    /// ```swift
    /// Tensor<Int>.broadcastShape([3, 1], [1, 4])  // [3, 4]
    /// Tensor<Int>.broadcastShape([3], [2, 3])      // [2, 3]
    /// Tensor<Int>.broadcastShape([3], [4])          // nil
    /// ```
    public static func broadcastShape(_ a: [Int], _ b: [Int]) -> [Int]? {
        let maxRank = Swift.max(a.count, b.count)
        var result = [Int](repeating: 0, count: maxRank)
        for i in 0..<maxRank {
            let dimA = i < a.count ? a[a.count - 1 - i] : 1
            let dimB = i < b.count ? b[b.count - 1 - i] : 1
            if dimA == dimB {
                result[maxRank - 1 - i] = dimA
            } else if dimA == 1 {
                result[maxRank - 1 - i] = dimB
            } else if dimB == 1 {
                result[maxRank - 1 - i] = dimA
            } else {
                return nil
            }
        }
        return result
    }

    /// Returns a view broadcast to `targetShape` using stride-0 for expanded dimensions.
    ///
    /// Dimensions of size 1 in the original shape that map to a larger size in
    /// `targetShape` get stride 0, which causes the single element to repeat
    /// across that dimension.
    ///
    /// - Parameter targetShape: The desired output shape. Must be broadcast-compatible.
    /// - Returns: A non-contiguous view with the target shape.
    public func broadcast(to targetShape: [Int]) -> Tensor {
        let rankDiff = targetShape.count - shape.count
        let paddedShape = [Int](repeating: 1, count: rankDiff) + shape
        let paddedStrides = [Int](repeating: 0, count: rankDiff) + strides
        var newStrides = [Int](repeating: 0, count: targetShape.count)
        for i in 0..<targetShape.count {
            newStrides[i] = paddedShape[i] == targetShape[i] ? paddedStrides[i] : 0
        }
        return Tensor(storage: storage, shape: targetShape, strides: newStrides,
                      offset: offset, isContiguous: false)
    }
}
