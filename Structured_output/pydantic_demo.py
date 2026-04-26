# EmailStr is a special type provided by Pydantic that validates whether the input string is a valid email address. If the input does not conform to the standard email format, Pydantic will raise a validation error. This ensures that any data assigned to the email field in the Student model is a properly formatted email address.

# Feilds in Pydantic models can have default values, which are used when no value is provided during instantiation. In the Student model, the name field has a default value of "Nehraj". This means that if you create an instance of Student without providing a name, it will automatically be set to "Nehraj". For example, if you instantiate Student with only age and email, the name will still be "Nehraj" by default.
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Nehraj"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0.0, lt=5.0, default=0.24, description="CGPA must be between 0.0 and 5.0")

Student = Student(name="Nehraj", age=25, email="nehraj@example.com")

# The model_dump() method in Pydantic is used to convert a Pydantic model instance into a dictionary.
student_dict = Student.model_dump()
# print(type(student_dict))

print(student_dict['name'])