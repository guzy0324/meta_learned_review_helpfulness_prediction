from argparse import ArgumentParser

from .switch import Switch

add_args_func = Switch()

add_group = True

def add_args(key: str, parser: ArgumentParser, **kwargs):
    global add_group
    if add_group:
        parser = parser.add_argument_group(key)
        add_group = False
        add_args_func[key](parser, **kwargs)
        add_group = True
    else:
        add_args_func[key](parser, **kwargs)

if __name__ == '__main__':

    @add_args_func("a")
    def a(parser):
        parser.add_argument("a")

    @add_args_func("b")
    def b(parser):
        a(parser)
        parser.add_argument("b")

    @add_args_func("c")
    def c(parser):
        parser.add_argument("c")

    parser = ArgumentParser()
    add_args("b", parser)
    add_args("c", parser)
    parser.parse_args()
