#if canImport(Accelerate)

import Testing
@testable import SwiftMatrix

struct TensorAccelerateAddTests {
    @Test func addFloat() {
        let a = Tensor(shape: [3], elements: [1.0, 2.0, 3.0] as [Float])
        let b = Tensor(shape: [3], elements: [4.0, 5.0, 6.0] as [Float])
        #expect(a + b == Tensor(shape: [3], elements: [5.0, 7.0, 9.0] as [Float]))
    }

    @Test func addDouble() {
        let a = Tensor(shape: [3], elements: [1.0, 2.0, 3.0])
        let b = Tensor(shape: [3], elements: [4.0, 5.0, 6.0])
        #expect(a + b == Tensor(shape: [3], elements: [5.0, 7.0, 9.0]))
    }

    @Test func addRank2Float() {
        let a = Tensor([[1.0, 2.0], [3.0, 4.0]] as [[Float]])
        let b = Tensor([[5.0, 6.0], [7.0, 8.0]] as [[Float]])
        #expect(a + b == Tensor([[6.0, 8.0], [10.0, 12.0]] as [[Float]]))
    }
}

struct TensorAccelerateSubTests {
    @Test func subFloat() {
        let a = Tensor(shape: [3], elements: [10.0, 20.0, 30.0] as [Float])
        let b = Tensor(shape: [3], elements: [1.0, 2.0, 3.0] as [Float])
        #expect(a - b == Tensor(shape: [3], elements: [9.0, 18.0, 27.0] as [Float]))
    }

    @Test func subDouble() {
        let a = Tensor(shape: [3], elements: [10.0, 20.0, 30.0])
        let b = Tensor(shape: [3], elements: [1.0, 2.0, 3.0])
        #expect(a - b == Tensor(shape: [3], elements: [9.0, 18.0, 27.0]))
    }
}

struct TensorAccelerateMulTests {
    @Test func mulFloat() {
        let a = Tensor(shape: [3], elements: [2.0, 3.0, 4.0] as [Float])
        let b = Tensor(shape: [3], elements: [5.0, 6.0, 7.0] as [Float])
        #expect(a * b == Tensor(shape: [3], elements: [10.0, 18.0, 28.0] as [Float]))
    }

    @Test func mulDouble() {
        let a = Tensor(shape: [3], elements: [2.0, 3.0, 4.0])
        let b = Tensor(shape: [3], elements: [5.0, 6.0, 7.0])
        #expect(a * b == Tensor(shape: [3], elements: [10.0, 18.0, 28.0]))
    }
}

struct TensorAccelerateDivTests {
    @Test func divFloat() {
        let a = Tensor(shape: [3], elements: [10.0, 20.0, 30.0] as [Float])
        let b = Tensor(shape: [3], elements: [2.0, 5.0, 6.0] as [Float])
        #expect(a / b == Tensor(shape: [3], elements: [5.0, 4.0, 5.0] as [Float]))
    }

    @Test func divDouble() {
        let a = Tensor(shape: [3], elements: [10.0, 20.0, 30.0])
        let b = Tensor(shape: [3], elements: [2.0, 5.0, 6.0])
        #expect(a / b == Tensor(shape: [3], elements: [5.0, 4.0, 5.0]))
    }
}

struct TensorAccelerateScalarTests {
    @Test func addScalarFloat() {
        let a = Tensor(shape: [3], elements: [1.0, 2.0, 3.0] as [Float])
        #expect(a + 10.0 == Tensor(shape: [3], elements: [11.0, 12.0, 13.0] as [Float]))
    }

    @Test func scalarAddFloat() {
        let a = Tensor(shape: [3], elements: [1.0, 2.0, 3.0] as [Float])
        #expect(10.0 + a == Tensor(shape: [3], elements: [11.0, 12.0, 13.0] as [Float]))
    }

    @Test func mulScalarDouble() {
        let a = Tensor(shape: [3], elements: [1.0, 2.0, 3.0])
        #expect(a * 3.0 == Tensor(shape: [3], elements: [3.0, 6.0, 9.0]))
    }

    @Test func scalarMulDouble() {
        let a = Tensor(shape: [3], elements: [1.0, 2.0, 3.0])
        #expect(3.0 * a == Tensor(shape: [3], elements: [3.0, 6.0, 9.0]))
    }

    @Test func divScalarFloat() {
        let a = Tensor(shape: [2], elements: [10.0, 20.0] as [Float])
        #expect(a / 2.0 == Tensor(shape: [2], elements: [5.0, 10.0] as [Float]))
    }

    @Test func subScalarDouble() {
        let a = Tensor(shape: [3], elements: [10.0, 20.0, 30.0])
        #expect(a - 1.0 == Tensor(shape: [3], elements: [9.0, 19.0, 29.0]))
    }

    @Test func scalarSubDouble() {
        let a = Tensor(shape: [3], elements: [1.0, 2.0, 3.0])
        #expect(10.0 - a == Tensor(shape: [3], elements: [9.0, 8.0, 7.0]))
    }
}

struct TensorAccelerateNegTests {
    @Test func negFloat() {
        let a = Tensor(shape: [3], elements: [1.0, -2.0, 3.0] as [Float])
        #expect(-a == Tensor(shape: [3], elements: [-1.0, 2.0, -3.0] as [Float]))
    }

    @Test func negDouble() {
        let a = Tensor(shape: [3], elements: [1.0, -2.0, 3.0])
        #expect(-a == Tensor(shape: [3], elements: [-1.0, 2.0, -3.0]))
    }
}

struct TensorAccelerateNonContiguousArithmeticTests {
    @Test func addTransposedFloat() {
        let a = Tensor([[1.0, 2.0], [3.0, 4.0]] as [[Float]]).transposed()
        let b = Tensor([[10.0, 20.0], [30.0, 40.0]] as [[Float]]).transposed()
        let result = a + b
        // a transposed: [[1,3],[2,4]], b transposed: [[10,30],[20,40]]
        // sum: [[11,33],[22,44]]
        let expected = Tensor(shape: [2, 2], elements: [11.0, 33.0, 22.0, 44.0] as [Float])
        #expect(result == expected)
    }

    @Test func mulTransposedDouble() {
        let a = Tensor([[1.0, 2.0], [3.0, 4.0]]).transposed()
        let b = Tensor(shape: [2, 2], elements: [1.0, 1.0, 1.0, 1.0])
        let result = a * b
        // a transposed iterates as [1,3,2,4]
        #expect(Array(result) == [1.0, 3.0, 2.0, 4.0])
    }
}

struct TensorAccelerateArithmeticConsistencyTests {
    @Test func accelerateMatchesGenericAdd() {
        let intA = Tensor(shape: [4], elements: [1, 2, 3, 4])
        let intB = Tensor(shape: [4], elements: [5, 6, 7, 8])
        let intResult = Array(intA + intB)

        let dblA = Tensor(shape: [4], elements: [1.0, 2.0, 3.0, 4.0])
        let dblB = Tensor(shape: [4], elements: [5.0, 6.0, 7.0, 8.0])
        let dblResult = Array(dblA + dblB)

        #expect(dblResult == intResult.map(Double.init))
    }

    @Test func accelerateMatchesGenericMul() {
        let intA = Tensor(shape: [3], elements: [2, 3, 4])
        let intB = Tensor(shape: [3], elements: [5, 6, 7])
        let intResult = Array(intA * intB)

        let fltA = Tensor(shape: [3], elements: [2.0, 3.0, 4.0] as [Float])
        let fltB = Tensor(shape: [3], elements: [5.0, 6.0, 7.0] as [Float])
        let fltResult = Array(fltA * fltB)

        #expect(fltResult == intResult.map(Float.init))
    }
}

#endif
