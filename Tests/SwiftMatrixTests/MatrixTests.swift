import Testing
@testable import SwiftMatrix

struct MatrixCreationTests {
    @Test func createFromNestedArrays() {
        let m = Matrix([[1, 2, 3], [4, 5, 6]])
        #expect(m.size == Vector(width: 3, depth: 2))
    }

    @Test func createFromSizeAndRepeatingValue() {
        let m = Matrix(size: Vector(width: 3, depth: 2), repeating: 0)
        #expect(m.size == Vector(width: 3, depth: 2))
        #expect(m[Pair(x: 0, y: 0)] == 0)
        #expect(m[Pair(x: 1, y: 2)] == 0)
    }
}

struct MatrixSubscriptTests {
    @Test func subscriptAccess() {
        let m = Matrix([[1, 2, 3], [4, 5, 6]])
        #expect(m[Pair(x: 0, y: 0)] == 1)
        #expect(m[Pair(x: 0, y: 2)] == 3)
        #expect(m[Pair(x: 1, y: 0)] == 4)
        #expect(m[Pair(x: 1, y: 2)] == 6)
    }
}

struct MatrixCollectionTests {
    @Test func iterationOrder() {
        let m = Matrix([[1, 2], [3, 4]])
        let elements = Array(m)
        #expect(elements == [1, 2, 3, 4])
    }

    @Test func startAndEndIndex() {
        let m = Matrix([[1, 2, 3], [4, 5, 6]])
        #expect(m.startIndex == Pair(x: 0, y: 0))
        #expect(m.endIndex == Pair(x: 2, y: 0))
    }
}

struct MatrixEquatableTests {
    @Test func equalMatrices() {
        let a = Matrix([[1, 2], [3, 4]])
        let b = Matrix([[1, 2], [3, 4]])
        #expect(a == b)
    }

    @Test func unequalMatrices() {
        let a = Matrix([[1, 2], [3, 4]])
        let b = Matrix([[1, 2], [3, 5]])
        #expect(a != b)
    }
}

struct CollectionViewTests {
    @Test func invertSwapsAxes() {
        let m = Matrix([[1, 2, 3], [4, 5, 6]])
        let inverted = m.invert()
        // Original [0,0]=1, [0,1]=2, [0,2]=3, [1,0]=4, [1,1]=5, [1,2]=6
        // Inverted swizzles (x,y) -> (y,x), so inverted[0,0] reads m[0,0]=1
        // inverted[0,1] reads m[1,0]=4, inverted[1,0] reads m[0,1]=2
        #expect(inverted[Pair(x: 0, y: 0)] == 1)
        #expect(inverted[Pair(x: 0, y: 1)] == 4)
        #expect(inverted[Pair(x: 1, y: 0)] == 2)
    }

    @Test func offsetWraps() {
        let m = Matrix([[1, 2], [3, 4]])
        let shifted = m.offset(by: Pair(x: 1, y: 1))
        // offset(by: (1,1)) means shifted[0,0] reads m[(0+1)%2, (0+1)%2] = m[1,1] = 4
        #expect(shifted[Pair(x: 0, y: 0)] == 4)
        // shifted[0,1] reads m[(0+1)%2, (1+1)%2] = m[1,0] = 3
        #expect(shifted[Pair(x: 0, y: 1)] == 3)
    }
}
