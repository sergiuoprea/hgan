__version__ = '0.1'
__author__ = 'Sergiu Oprea'

"""
Calculating the Frechet Inception Distance (FID) to evaluate our CycleGAN.

In contrast with Inception Distance metric, FID operates comparing the statistics of generated to
real samples. It consists on the Frechét distance between multivariate Gaussians X_ with given mean
and covariance (mu, C) representing the distribution of generated (_g) and real data (_r). A lower
FID results is better and corresponds to more similarity between the generated and real samples. The
mean and covariance are obtained from the activations extracted from the last pooling layer of the
Inception v3 model (2048 features per sample).

The FID distance between two multivariate Gaussians X_r ~ N(mu_r, C_r) and X_g ~ N(mu_g, C_g) is:

            d^2 = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 * sqrt(C_r * C_g)) where,

- mu_r and mu_g are feature-wise mean of the real and generated data.
- C_r and C_g are the covariance matrices for the feature vectors
- Tr is the trace linear algebra equation (sum of elements along the main diagonal of the square
  matrix)

Important notes:
- The number of samples to calculate the Gaussian statistics (mean and covariance) should be greater
than the dimension of the coding layer (e.g. using pool 3 layer of Inception v3 network, the dimen-
sion would be 2048). Otherwise the covariance is not full rankl resulting in complex numbers and
nans by calculating the square root.
- A recommendation is the usage of a minumum sample size of 10.000 to calculate the FID to not
underestimate the true FID of the generator.

Original paper where FID metric was proposed: https://arxiv.org/abs/1706.08500
Official Tensorflow implementation: https://github.com/bioinf-jku/TTUR
Unofficial Pytorch implementation: https://github.com/mseitzer/pytorch-fid - Note that due to the
differences in the TF and Pytorch backends FID result will be slightly different from the original
implementation (e.g. .08 absolute error and 0.0009 relative error on LSUN, using ProGAN generated
images).

This implementation is based on the original paper and on the aforementioned implementations in
PyTorch and Tensorflow.
"""

# Standard library imports
import numpy as np
import scipy.linalg as linalg

def _calculate_data_statistics(data: list):
    """ Calculating the statistics used by FID metric.

     Parameters:
    -- data         : List of feature vectors.

    Returns:
    -- Mean and covariance matrix of input data
    """
    #if not isinstance(data, list): data = [data]

    _mu = np.mean(data, axis=0)
    _sigma = np.cov(data, rowvar=False) # Covariance matrix taking each column as a variable

    return _mu, _sigma


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    Stable version by Dougal J. Sutherland and implemented in
    https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py

    Parameters:
    -- mu1   : mean over samples of the activations of the pool_3 layer of
               the inception model for generated or real samples.
    -- mu2   : mean over samples of the activations of the pool_3 layer of
               the inception model of an representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    -- eps   : offset to avoid singular product (default: 1e-6).
    Returns:
    --  The Frechet Distance.
    """

    # Checklist before takeoff
    _mu1 = np.atleast_1d(mu1)
    _mu2 = np.atleast_1d(mu2)

    _sigma1 = np.atleast_2d(sigma1)
    _sigma2 = np.atleast_2d(sigma2)

    assert _mu1.shape == _mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert _sigma1.shape == _sigma2.shape, \
        'Training and test covariances have different dimensions'

    # Sum of the squared differences ||mu1 - mu2||^2
    _mu_diff = mu1 - mu2
    _mu_ssd = _mu_diff.dot(_mu_diff)

    # Product might be almost singular
    # Square root of the covariance matrices product
    _covmean, _ = linalg.sqrtm(_sigma1.dot(_sigma2), disp=False)

    if not np.isfinite(_covmean).all():
        msg = (f'fid calculation produces singular product; '
               'adding {eps} to diagonal of cov estimates')
        print(msg)
        _offset = np.eye(_sigma1.shape[0]) * eps
        _covmean = linalg.sqrtm((_sigma1 + _offset).dot(_sigma2 + _offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(_covmean):
        if not np.allclose(np.diagonal(_covmean).imag, 0, atol=1e-3):
            _img_comp = np.max(np.abs(_covmean.imag))
            raise ValueError(f'Imaginary component {_img_comp}')
        _covmean = _covmean.real

    _tr_covmean = np.trace(_covmean)

    return _mu_ssd + np.trace(_sigma1) + np.trace(_sigma2) - 2 * _tr_covmean

def fid(features_1, features_2):
    """ Calculating Frechet Inception Distance between two data distributions

    Parameters:
    -- paths        : List containing the two folder paths corresponding to each distribution.
    -- model        : Instance of inception model.
    -- device       : Device used for the computation.
    -- batch_size   : Batch size of images for the model to process at once. The total number
                      of images should be multiple of the batch size (default: 50).
    -- dims         : Number of features extracted from the Inception net per each input
                      (default: 2048).

    Returns:
    -- The FID score/distance.
    """
    _m1, _s1 = _calculate_data_statistics(list(features_1))
    _m2, _s2 = _calculate_data_statistics(list(features_2))

    _score = _calculate_frechet_distance(mu1=_m1, sigma1=_s1, mu2=_m2, sigma2=_s2)

    return _score
