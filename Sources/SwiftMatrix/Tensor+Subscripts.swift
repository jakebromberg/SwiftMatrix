extension Tensor {
    public subscript(indices: [Int]) -> Element {
        get { storage[storageIndex(for: indices)] }
        set { storage[storageIndex(for: indices)] = newValue }
    }

    public subscript(indices: Int...) -> Element {
        get { storage[storageIndex(for: indices)] }
        set { storage[storageIndex(for: indices)] = newValue }
    }
}
