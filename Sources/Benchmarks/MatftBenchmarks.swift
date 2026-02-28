import Matft

/// Benchmark functions for Matft.
///
/// `MfArray` is constructed from flat arrays with explicit shape and `mftype`.
/// Matft uses class-based reference semantics and type-erased storage under the hood.
enum MatftBenchmarks {

    // MARK: - Float

    static func addFloat(a: [Float], b: [Float], n: Int) {
        let ma = MfArray(a, mftype: .Float, shape: [n, n])
        let mb = MfArray(b, mftype: .Float, shape: [n, n])
        blackHole(ma + mb)
    }

    static func matmulFloat(a: [Float], b: [Float], n: Int) {
        let ma = MfArray(a, mftype: .Float, shape: [n, n])
        let mb = MfArray(b, mftype: .Float, shape: [n, n])
        blackHole(Matft.matmul(ma, mb))
    }

    static func sumFloat(a: [Float], n: Int) {
        let ma = MfArray(a, mftype: .Float, shape: [n, n])
        blackHole(ma.sum())
    }

    static func dotFloat(a: [Float], b: [Float], n: Int) {
        let va = MfArray(a, mftype: .Float, shape: [n * n])
        let vb = MfArray(b, mftype: .Float, shape: [n * n])
        blackHole(Matft.inner(va, vb))
    }

    // MARK: - Double

    static func addDouble(a: [Double], b: [Double], n: Int) {
        let ma = MfArray(a, mftype: .Double, shape: [n, n])
        let mb = MfArray(b, mftype: .Double, shape: [n, n])
        blackHole(ma + mb)
    }

    static func matmulDouble(a: [Double], b: [Double], n: Int) {
        let ma = MfArray(a, mftype: .Double, shape: [n, n])
        let mb = MfArray(b, mftype: .Double, shape: [n, n])
        blackHole(Matft.matmul(ma, mb))
    }

    static func sumDouble(a: [Double], n: Int) {
        let ma = MfArray(a, mftype: .Double, shape: [n, n])
        blackHole(ma.sum())
    }

    static func dotDouble(a: [Double], b: [Double], n: Int) {
        let va = MfArray(a, mftype: .Double, shape: [n * n])
        let vb = MfArray(b, mftype: .Double, shape: [n * n])
        blackHole(Matft.inner(va, vb))
    }
}
