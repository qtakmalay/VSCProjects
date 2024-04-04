from bayesian_net import BayesianNet

T, F = True, False


def build_network():
    bn = BayesianNet([('VA', '', 0.09),
                      ('MP', '', 0.01),
                      ('SI', 'MP', {T: 0.5, F: 0.05}),
                      ('RE', 'VA', {T: 0.85, F: 0.4}),
                      ('HA', 'RE SI', {(T, T): 0.9, (T, F): 0.65, (F, T): 0.2, (F, F): 0.03}),
                      ('DA', 'MP', {T: 0.3, F: 0.05})])

    # TODO: build network that is provided for you on Moodle: A5 Exact Inference, include probability distributions

    return bn


if __name__ == '__main__':
    bn = build_network()
    # optional: visualize network to check whether the structure is correct

    # TODO: compute the answers to the probabilistic queries (provided on Moodle A5 Exact Inference)
    # TODO: print results
    # TODO: enter required numbers in Moodle
    # Hint: use bn.event_probability(event)
    alpha = bn.event_probability({'MP': T, 'VA': T, 'RE': T, 'HA': T})
    p2 = bn.event_probability({'MP': T, 'SI': T, 'HA': T, 'VA': T, 'RE': T})
    p3 = bn.event_probability({'MP': T, 'SI': F, 'HA': T, 'VA': T, 'RE': T})
    p4 = bn.event_probability({'MP': T, 'VA': T, 'RE': T, 'HA': T, 'SI': T}) / bn.event_probability(
        {'MP': T, 'VA': T, 'RE': T, 'HA': T})
    p5 = bn.event_probability({'MP': T, 'VA': T, 'RE': T, 'HA': T, 'SI': F}) / bn.event_probability(
        {'MP': T, 'VA': T, 'RE': T, 'HA': T})
    print('alpha =', alpha)
    print('P(mp, si, ha, va, re) =', p2)
    print('P(mp, ¬ si, ha, va, re) =', p3)
    print('P(si | mp, va, re, ha) =', p4)
    print('P(¬ si | mp, va, re, ha) =', p5)


