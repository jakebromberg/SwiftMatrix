import Surge

/// Benchmark functions for Surge.
///
/// Surge's `Matrix<Scalar>` uses `*` for matrix multiplication (GEMM) and `+` for
/// element-wise addition. For sum, we use Surge's free `sum()` function on flat arrays since
/// `Matrix.grid` is internal and `sum(matrix, axies:)` returns a matrix, not a scalar.
enum SurgeBenchmarks {

    // MARK: - Float

    static func addFloat(a: [Float], b: [Float], n: Int) {
        let ma = Matrix<Float>(rows: n, columns: n, grid: a)
        let mb = Matrix<Float>(rows: n, columns: n, grid: b)
        blackHole(ma + mb)
    }

    static func matmulFloat(a: [Float], b: [Float], n: Int) {
        let ma = Matrix<Float>(rows: n, columns: n, grid: a)
        let mb = Matrix<Float>(rows: n, columns: n, grid: b)
        blackHole(ma * mb)
    }

    static func sumFloat(a: [Float], n: Int) {
        blackHole(Surge.sum(a))
    }

    static func dotFloat(a: [Float], b: [Float], n: Int) {
        blackHole(Surge.dot(a, b))
    }

    // MARK: - Double

    static func addDouble(a: [Double], b: [Double], n: Int) {
        let ma = Matrix<Double>(rows: n, columns: n, grid: a)
        let mb = Matrix<Double>(rows: n, columns: n, grid: b)
        blackHole(ma + mb)
    }

    static func matmulDouble(a: [Double], b: [Double], n: Int) {
        let ma = Matrix<Double>(rows: n, columns: n, grid: a)
        let mb = Matrix<Double>(rows: n, columns: n, grid: b)
        blackHole(ma * mb)
    }

    static func sumDouble(a: [Double], n: Int) {
        blackHole(Surge.sum(a))
    }

    static func dotDouble(a: [Double], b: [Double], n: Int) {
        blackHole(Surge.dot(a, b))
    }
}
