from bayesian_net import BayesianNet

bn = BayesianNet([('Colourful', ''),
    ('Energy', ''),
    ('Month', ''),
    ('Weather', 'Month'),
    ('Vicinity', ''),
    ('Flight', 'Energy Weather Vicinity '),

])


# TODO: visualize the result
# TODO: include your first name and matriculation number in the title
bn.draw('Stepanov Sergey k12140199')


