import minitorch
import pytest
from minitorch import History, Variable, Scalar

one_arg = [
    ("neg", lambda a: -a),
    ("addconstant", lambda a: a + 5),
    ("subconstant", lambda a: a - 5),
    ("mult", lambda a: 5 * a),
    ("div", lambda a: a / 5),
    ("sig", lambda a: a.sigmoid(), lambda a: minitorch.operators.sigmoid(a)),
    (
        "log",
        lambda a: (a + 100000).log(),
        lambda a: minitorch.operators.log(a + 100000),
    ),
    ("relu", lambda a: (a + 5.5).relu(), lambda a: minitorch.operators.relu(a + 5.5)),
]


def test_one_derivative(fn, t1):
    minitorch.derivative_check(fn[1], t1)


two_arg = [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / (b + 5.5)),
]


def test_two_derivative(fn, t1, t2):
    minitorch.derivative_check(fn[1], t1, t2)
print(two_arg[2])
test_two_derivative(two_arg[2], t1=Scalar(0.000000), t2=Scalar(0.000000))
