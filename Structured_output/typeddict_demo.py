from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

person1: Person = {"name": "Alice", "age": 30}
person2: Person = {"name": "Bob", "age": '25'} # This won't raise an error at runtime, but it violates the type hint and would be flagged by static type checkers like mypy.
print(person1)
print(person2)