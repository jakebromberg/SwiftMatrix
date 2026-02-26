import Testing
@testable import SwiftMatrix

struct TensorDotProductTests {
    @Test func dotProductRank1() {
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor(shape: [3], elements: [4, 5, 6])
        #expect(Tensor.dot(a, b) == 32) // 1*4 + 2*5 + 3*6
    }

    @Test func dotProductSingleElement() {
        let a = Tensor(shape: [1], elements: [3])
        let b = Tensor(shape: [1], elements: [4])
        #expect(Tensor.dot(a, b) == 12)
    }
}

struct TensorMatmulTests {
    @Test func matmul2x3Times3x2() {
        let a = Tensor([[1, 2, 3], [4, 5, 6]])         // 2x3
        let b = Tensor([[7, 8], [9, 10], [11, 12]])     // 3x2
        let c = Tensor<Int>.matmul(a, b)
        #expect(c.shape == [2, 2])
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        #expect(c == Tensor([[58, 64], [139, 154]]))
    }

    @Test func matmulIdentity() {
        let a = Tensor([[1, 2], [3, 4]])
        let identity = Tensor([[1, 0], [0, 1]])
        #expect(Tensor<Int>.matmul(a, identity) == a)
    }
}

struct TensorSumTests {
    @Test func sumAll() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]])
        #expect(t.sum() == 21)
    }

    @Test func sumAxis0() {
        // Sum rows: [[1,2,3],[4,5,6]] -> [5,7,9]
        let t = Tensor([[1, 2, 3], [4, 5, 6]])
        let s = t.sum(axis: 0)
        #expect(s.shape == [3])
        #expect(s == Tensor(shape: [3], elements: [5, 7, 9]))
    }

    @Test func sumAxis1() {
        // Sum columns: [[1,2,3],[4,5,6]] -> [6,15]
        let t = Tensor([[1, 2, 3], [4, 5, 6]])
        let s = t.sum(axis: 1)
        #expect(s.shape == [2])
        #expect(s == Tensor(shape: [2], elements: [6, 15]))
    }

    @Test func sumAxis0NonContiguous() {
        // Transposed 2x3 -> 3x2 with strides [1,3]
        // Logical elements: [1,4,2,5,3,6]
        let t = Tensor(
            storage: [1, 2, 3, 4, 5, 6],
            shape: [3, 2],
            strides: [1, 3],
            offset: 0,
            isContiguous: false
        )
        // sum axis 0: [1+2+3, 4+5+6] = [6, 15]
        let s = t.sum(axis: 0)
        #expect(s.shape == [2])
        #expect(s == Tensor(shape: [2], elements: [6, 15]))
    }
}

struct TensorMeanTests {
    @Test func meanAll() {
        let t = Tensor(shape: [4], elements: [1.0, 2.0, 3.0, 4.0])
        #expect(t.mean() == 2.5)
    }

    @Test func meanAxis0() {
        // Mean across rows: [[1,2,3],[4,5,6]] -> [2.5, 3.5, 4.5]
        let t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let m = t.mean(axis: 0)
        #expect(m.shape == [3])
        #expect(m == Tensor(shape: [3], elements: [2.5, 3.5, 4.5]))
    }

    @Test func meanAxis1() {
        // Mean across columns: [[1,2,3],[4,5,6]] -> [2.0, 5.0]
        let t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let m = t.mean(axis: 1)
        #expect(m.shape == [2])
        #expect(m == Tensor(shape: [2], elements: [2.0, 5.0]))
    }
}
