from pytreeclass import tree_viz


def test_vbox():

    assert tree_viz.vbox('a', ' a',
                         'a ') == '┌──┐\n│a │\n├──┤\n│ a│\n├──┤\n│a │\n└──┘'


def test_hbox():
    assert tree_viz.hbox('a', 'b', 'c') == '┌─┬─┬─┐\n│a│b│c│\n└─┴─┴─┘\n'
