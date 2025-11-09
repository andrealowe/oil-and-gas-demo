def test_functionality():
    # A simple test function that checks basic operations
    assert 1 + 1 == 2, "Test failed: 1 + 1 should equal 2"
    assert "Hello" + " World" == "Hello World", "Test failed: String concatenation failed"
    assert len([1, 2, 3]) == 3, "Test failed: Length of list should be 3"

if __name__ == "__main__":
    test_functionality()
    print("All tests passed!")