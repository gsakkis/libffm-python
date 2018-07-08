import logging
from click import group, command, option, argument, File, Path
from ffm import FFM, read_libffm


@group(context_settings={
    'help_option_names': ['-h', '--help'],
    'max_content_width': 120,
})
@option('-q', '--quiet', 'log_level', flag_value=logging.WARN)
@option('-v', '--verbose', 'log_level', flag_value=logging.DEBUG, default=True)
def cli(log_level):
    logging.basicConfig(level=log_level, format='%(message)s')


@cli.command()
@argument('training_file', type=Path(exists=True, dir_okay=False))
@argument('model_file', required=False, type=Path(writable=True, dir_okay=False))
@option('-l', '--lambda', type=float, default=0.00002,
        help='Regularization parameter')
@option('-k', '--factors', type=int, default=4,
        help='Number of latent factors')
@option('-r', '--eta', type=float, default=0.2,
        help='Learning rate')
@option('-t', '--iterations', type=int, default=15,
        help='Number of iterations')
@option('-p', '--validation-file', type=Path(exists=True, dir_okay=False),
        help='Path to validation set')
@option('-a', '--auto-stop', type=int, default=0, metavar='N',
        help='Keep iterating at most <N> times without achieving a better validation score')
@option('--score', default='neg_log_loss',
        help='Metric to use for evaluating performance on validation set. '
             'It can be any predefined sklearn scoring metric '
             '(http://scikit-learn.org/stable/modules/model_evaluation.html)')
@option('-s', '--threads', type=int, default=1,
        help='Number of threads')
@option('--norm/--no-norm', default=True,
        help='enable/disable instance-wise normalization')
@option('--rand/--no-rand', default=True,
        help='enable/disable randomization')
@option('--bin/--no-bin', default=True,
        help='enable/disable binary file generation from training/validation files')
def train(training_file, validation_file, **kwargs):
    """Train a FFM model"""
    model = FFM(
        lam=kwargs['lambda'],
        k=kwargs['factors'],
        eta=kwargs['eta'],
        nr_iters=kwargs['iterations'],
        auto_stop=kwargs['auto_stop'],
        scorer=kwargs['score'],
        nr_threads=kwargs['threads'],
        normalization=kwargs['norm'],
        randomization=kwargs['rand'],
    )
    if kwargs['bin']:
        model.fit_from_file(training_file, validation_file)
    else:
        train_X, train_y = read_libffm(training_file)
        val_X_y = read_libffm(validation_file) if validation_file else None
        model.fit(train_X, train_y, val_X_y)
    model.save_model(kwargs['model_file'] or training_file + '.model')


@cli.command()
@argument('model_file', type=Path(exists=True, dir_okay=False))
@argument('test_file', type=Path(exists=True, dir_okay=False))
@argument('output_file', type=File('w'), required=False)
def predict(model_file, test_file, output_file):
    """Use a trained FFM model to make predictions"""
    model = FFM().read_model(model_file)
    test_X, test_y = read_libffm(test_file)
    for p in model.predict_proba(test_X):
        print('{:.6g}'.format(p), file=output_file)


if __name__ == '__main__':
    cli()
