#if canImport(Accelerate)

import Testing
@testable import SwiftMatrix

// MARK: - Sum

struct TensorAccelerateSumTests {
    @Test func sumFloat() {
        let t = Tensor(shape: [6], elements: [1, 2, 3, 4, 5, 6] as [Float])
        #expect(t.sum() == 21)
    }

    @Test func sumDouble() {
        let t = Tensor(shape: [6], elements: [1, 2, 3, 4, 5, 6] as [Double])
        #expect(t.sum() == 21)
    }

    @Test func sumNonContiguousFloat() {
        // Transposed 2x3 -> 3x2 view (non-contiguous)
        let t = Tensor(
            storage: [1, 2, 3, 4, 5, 6] as [Float],
            shape: [3, 2],
            strides: [1, 3],
            offset: 0,
            isContiguous: false
        )
        // Logical elements: [1, 4, 2, 5, 3, 6] -> sum = 21
        #expect(t.sum() == 21)
    }

    @Test func sumAxis0Float() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]] as [[Float]])
        let s = t.sum(axis: 0)
        #expect(s.shape == [3])
        #expect(s == Tensor(shape: [3], elements: [5, 7, 9] as [Float]))
    }

    @Test func sumAxis1Float() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]] as [[Float]])
        let s = t.sum(axis: 1)
        #expect(s.shape == [2])
        #expect(s == Tensor(shape: [2], elements: [6, 15] as [Float]))
    }

    @Test func sumSingleElementFloat() {
        let t = Tensor(shape: [1], elements: [42] as [Float])
        #expect(t.sum() == 42)
    }
}

// MARK: - Mean

struct TensorAccelerateMeanTests {
    @Test func meanFloat() {
        let t = Tensor(shape: [4], elements: [1, 2, 3, 4] as [Float])
        #expect(t.mean() == 2.5)
    }

    @Test func meanDouble() {
        let t = Tensor(shape: [4], elements: [1, 2, 3, 4] as [Double])
        #expect(t.mean() == 2.5)
    }

    @Test func meanAxis0Float() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]] as [[Float]])
        let m = t.mean(axis: 0)
        #expect(m.shape == [3])
        #expect(m == Tensor(shape: [3], elements: [2.5, 3.5, 4.5] as [Float]))
    }

    @Test func meanAxis1Float() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]] as [[Float]])
        let m = t.mean(axis: 1)
        #expect(m.shape == [2])
        #expect(m == Tensor(shape: [2], elements: [2, 5] as [Float]))
    }
}

// MARK: - Dot Product

struct TensorAccelerateDotTests {
    @Test func dotFloat() {
        let a = Tensor(shape: [3], elements: [1, 2, 3] as [Float])
        let b = Tensor(shape: [3], elements: [4, 5, 6] as [Float])
        #expect(Tensor.dot(a, b) == 32)
    }

    @Test func dotDouble() {
        let a = Tensor(shape: [3], elements: [1, 2, 3] as [Double])
        let b = Tensor(shape: [3], elements: [4, 5, 6] as [Double])
        #expect(Tensor.dot(a, b) == 32)
    }
}

// MARK: - Matmul

struct TensorAccelerateMatmulTests {
    @Test func matmulFloat() {
        let a = Tensor([[1, 2, 3], [4, 5, 6]] as [[Float]])     // 2x3
        let b = Tensor([[7, 8], [9, 10], [11, 12]] as [[Float]]) // 3x2
        let c = Tensor<Float>.matmul(a, b)
        #expect(c.shape == [2, 2])
        #expect(c == Tensor([[58, 64], [139, 154]] as [[Float]]))
    }

    @Test func matmulDouble() {
        let a = Tensor([[1, 2, 3], [4, 5, 6]] as [[Double]])     // 2x3
        let b = Tensor([[7, 8], [9, 10], [11, 12]] as [[Double]]) // 3x2
        let c = Tensor<Double>.matmul(a, b)
        #expect(c.shape == [2, 2])
        #expect(c == Tensor([[58, 64], [139, 154]] as [[Double]]))
    }

    @Test func matmulIdentityFloat() {
        let a = Tensor([[1, 2], [3, 4]] as [[Float]])
        let identity = Tensor([[1, 0], [0, 1]] as [[Float]])
        #expect(Tensor<Float>.matmul(a, identity) == a)
    }

    @Test func matmulNonContiguousFloat() {
        // Transpose a 2x3 matrix to get a non-contiguous 3x2, then matmul
        let a = Tensor([[1, 2, 3], [4, 5, 6]] as [[Float]])
        let at = a.transposed() // 3x2, non-contiguous
        // at * a should be 3x3: [[17,22,27],[22,29,36],[27,36,45]]
        let c = Tensor<Float>.matmul(at, a)
        #expect(c.shape == [3, 3])
        #expect(c == Tensor([
            [17, 22, 27],
            [22, 29, 36],
            [27, 36, 45],
        ] as [[Float]]))
    }
}

#endif
