import Testing
@testable import SwiftMatrix

struct TensorTensorArithmeticTests {
    @Test func addition() {
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor(shape: [3], elements: [4, 5, 6])
        #expect(a + b == Tensor(shape: [3], elements: [5, 7, 9]))
    }

    @Test func subtraction() {
        let a = Tensor(shape: [3], elements: [4, 5, 6])
        let b = Tensor(shape: [3], elements: [1, 2, 3])
        #expect(a - b == Tensor(shape: [3], elements: [3, 3, 3]))
    }

    @Test func multiplication() {
        let a = Tensor(shape: [2], elements: [2, 3])
        let b = Tensor(shape: [2], elements: [4, 5])
        #expect(a * b == Tensor(shape: [2], elements: [8, 15]))
    }

    @Test func division() {
        let a = Tensor(shape: [2], elements: [10.0, 20.0])
        let b = Tensor(shape: [2], elements: [2.0, 5.0])
        #expect(a / b == Tensor(shape: [2], elements: [5.0, 4.0]))
    }

    @Test func negation() {
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        #expect(-a == Tensor(shape: [3], elements: [-1, -2, -3]))
    }
}

struct TensorScalarArithmeticTests {
    @Test func tensorPlusScalar() {
        let t = Tensor(shape: [3], elements: [1, 2, 3])
        #expect(t + 10 == Tensor(shape: [3], elements: [11, 12, 13]))
    }

    @Test func scalarPlusTensor() {
        let t = Tensor(shape: [3], elements: [1, 2, 3])
        #expect(10 + t == Tensor(shape: [3], elements: [11, 12, 13]))
    }

    @Test func tensorMinusScalar() {
        let t = Tensor(shape: [3], elements: [10, 20, 30])
        #expect(t - 1 == Tensor(shape: [3], elements: [9, 19, 29]))
    }

    @Test func scalarMinusTensor() {
        let t = Tensor(shape: [3], elements: [1, 2, 3])
        #expect(10 - t == Tensor(shape: [3], elements: [9, 8, 7]))
    }

    @Test func tensorTimesScalar() {
        let t = Tensor(shape: [3], elements: [1, 2, 3])
        #expect(t * 3 == Tensor(shape: [3], elements: [3, 6, 9]))
    }

    @Test func scalarTimesTensor() {
        let t = Tensor(shape: [3], elements: [1, 2, 3])
        #expect(3 * t == Tensor(shape: [3], elements: [3, 6, 9]))
    }

    @Test func tensorDividedByScalar() {
        let t = Tensor(shape: [2], elements: [10.0, 20.0])
        #expect(t / 2.0 == Tensor(shape: [2], elements: [5.0, 10.0]))
    }
}

struct TensorCompoundAssignmentTests {
    @Test func plusEquals() {
        var a = Tensor(shape: [3], elements: [1, 2, 3])
        a += Tensor(shape: [3], elements: [4, 5, 6])
        #expect(a == Tensor(shape: [3], elements: [5, 7, 9]))
    }

    @Test func minusEquals() {
        var a = Tensor(shape: [3], elements: [10, 20, 30])
        a -= Tensor(shape: [3], elements: [1, 2, 3])
        #expect(a == Tensor(shape: [3], elements: [9, 18, 27]))
    }

    @Test func timesEquals() {
        var a = Tensor(shape: [2], elements: [2, 3])
        a *= Tensor(shape: [2], elements: [4, 5])
        #expect(a == Tensor(shape: [2], elements: [8, 15]))
    }

    @Test func divideEquals() {
        var a = Tensor(shape: [2], elements: [10.0, 20.0])
        a /= Tensor(shape: [2], elements: [2.0, 5.0])
        #expect(a == Tensor(shape: [2], elements: [5.0, 4.0]))
    }

    @Test func plusEqualsNonContiguous() {
        // lhs is contiguous, rhs is transposed (non-contiguous)
        var a = Tensor(shape: [3, 2], elements: [1, 2, 3, 4, 5, 6])
        let b = Tensor([[1, 3, 5], [2, 4, 6]]).transposed() // shape [3,2], non-contiguous
        a += b
        #expect(a == Tensor(shape: [3, 2], elements: [2, 4, 6, 8, 10, 12]))
    }

    @Test func plusEqualsRank2Contiguous() {
        var a = Tensor([[1, 2], [3, 4]])
        a += Tensor([[10, 20], [30, 40]])
        #expect(a == Tensor([[11, 22], [33, 44]]))
    }
}

struct TensorArithmeticShapeTests {
    @Test func preservesShapeRank2() {
        let a = Tensor([[1, 2, 3], [4, 5, 6]])
        let b = Tensor([[7, 8, 9], [10, 11, 12]])
        let c = a + b
        #expect(c.shape == [2, 3])
        #expect(c == Tensor([[8, 10, 12], [14, 16, 18]]))
    }

    @Test func worksWithNonContiguous() {
        // Transposed 2x3 -> 3x2 with non-contiguous strides
        let t = Tensor(
            storage: [1, 2, 3, 4, 5, 6],
            shape: [3, 2],
            strides: [1, 3],
            offset: 0,
            isContiguous: false
        )
        let ones = Tensor(shape: [3, 2], repeating: 1)
        let result = t + ones
        // t iterates as [1, 4, 2, 5, 3, 6], so result is [2, 5, 3, 6, 4, 7]
        #expect(Array(result) == [2, 5, 3, 6, 4, 7])
    }
}
