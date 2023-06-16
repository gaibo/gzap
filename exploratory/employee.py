# https://realpython.com/python-classes/#understanding-the-benefits-of-using-classes-in-python
from datetime import datetime


class Employee:
    company = "Example, Inc."

    def __init__(self, name, birth_date):
        self.name = name
        self.birth_date = birth_date

    @property
    def birth_date(self):
        return self._birth_date

    @birth_date.setter
    def birth_date(self, value):
        self._birth_date = datetime.fromisoformat(value)

    def compute_age(self):
        today = datetime.today()
        age_overestimate = today.year - self.birth_date.year
        this_year_birthday = datetime(
            today.year,
            self.birth_date.month,
            self.birth_date.day
        )
        if today.date() == this_year_birthday.date():
            print("Happy Birthday!")
        if today < this_year_birthday:
            return age_overestimate - 1
        else:
            return age_overestimate

    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)

    def __str__(self):
        return f"{self.name} is {self.compute_age()} years old"

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"name='{self.name}', "
            f"birth_date='{self.birth_date.strftime('%Y-%m-%d')}')"
        )
