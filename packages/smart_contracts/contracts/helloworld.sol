pragma solidity ^0.8.10;

contract HelloWorld {
  uint256 _counter;

  function increment() public {
    _counter += 1;
  }

  function getCounter() public view returns(uint256){
    return _counter;
  }
}