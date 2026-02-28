import SwiftMatrix

/// Benchmark functions for SwiftMatrix.
///
/// Each function accepts pre-generated flat arrays and a matrix size, constructs
/// SwiftMatrix tensors, performs the operation, and consumes the result via `blackHole`.
enum SwiftMatrixBenchmarks {

    // MARK: - Float

    static func addFloat(a: [Float], b: [Float], n: Int) {
        let ta = Tensor<Float>(shape: [n, n], elements: a)
        let tb = Tensor<Float>(shape: [n, n], elements: b)
        blackHole(ta + tb)
    }

    static func matmulFloat(a: [Float], b: [Float], n: Int) {
        let ta = Tensor<Float>(shape: [n, n], elements: a)
        let tb = Tensor<Float>(shape: [n, n], elements: b)
        blackHole(Tensor.matmul(ta, tb))
    }

    static func sumFloat(a: [Float], n: Int) {
        let ta = Tensor<Float>(shape: [n, n], elements: a)
        blackHole(ta.sum())
    }

    static func dotFloat(a: [Float], b: [Float], n: Int) {
        let va = Tensor<Float>(shape: [n * n], elements: a)
        let vb = Tensor<Float>(shape: [n * n], elements: b)
        blackHole(Tensor.dot(va, vb))
    }

    // MARK: - Double

    static func addDouble(a: [Double], b: [Double], n: Int) {
        let ta = Tensor<Double>(shape: [n, n], elements: a)
        let tb = Tensor<Double>(shape: [n, n], elements: b)
        blackHole(ta + tb)
    }

    static func matmulDouble(a: [Double], b: [Double], n: Int) {
        let ta = Tensor<Double>(shape: [n, n], elements: a)
        let tb = Tensor<Double>(shape: [n, n], elements: b)
        blackHole(Tensor.matmul(ta, tb))
    }

    static func sumDouble(a: [Double], n: Int) {
        let ta = Tensor<Double>(shape: [n, n], elements: a)
        blackHole(ta.sum())
    }

    static func dotDouble(a: [Double], b: [Double], n: Int) {
        let va = Tensor<Double>(shape: [n * n], elements: a)
        let vb = Tensor<Double>(shape: [n * n], elements: b)
        blackHole(Tensor.dot(va, vb))
    }
}
