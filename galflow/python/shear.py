# Functions computing shear related affine transformations
# based on https://github.com/GalSim-developers/GalSim/blob/main/galsim/shear.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons import image as tfa_image
from galflow.python.transform import transform

__all__ = ["shear", "shear_transformation"]

def shear_transformation(g1, g2, Fourier=False, name=None):
  """Function to compute the affine transformation corresponding to a given shear.
  This function uses the reduced shear definition:

    :math:`|g| = \frac{a - b}{a + b}`

  If a field is sheared by some shear, s, then the position (x,y) -> (x',y')
  according to:
  .. math::
      \left( \begin{array}{c} x^\prime \\ y^\prime \end{array} \right)
      = S \left( \begin{array}{c} x \\ y \end{array} \right)
  and :math:`S` is the return value of this function ``S = shear.getMatrix()``.
  Specifically, the matrix is
  .. math::
      S = \frac{1}{\sqrt{1-g^2}}
              \left( \begin{array}{cc} 1+g_1 & g_2 \\
                                       g_2 & 1-g_1 \end{array} \right)
  Args: 
    g1: `Tensor`, The first component of the shear in the "reduced shear" definition.
    g2: `Tensor`, The second component of the shear in the "reduced shear" definition.
    Fourier: `boolean`, when doing an interpolation in Fourier space, the center of pixels is on
    integer values, so set to true if transforming in Fourier space.
    name: `string`, name of the operation.
    
  Returns:
    `Tensor` of the transformation matrix of shape [(batch), 3, 3]
  """

  with tf.name_scope(name or "shear_transformation"):
    g1 = tf.convert_to_tensor(g1, dtype=tf.float32)
    g2 = tf.convert_to_tensor(g2, dtype=tf.float32)
    gsqr = g1**2 + g2**2

    # Building a batched jacobian
    jac = tf.stack([ 1. + g1, g2,
                  g2, 1. - g1], axis=1) / tf.expand_dims(tf.sqrt(1.- gsqr),1)
    jac = tf.reshape(jac, [-1,2,2])

    # Inverting these jacobians to follow the TF definition
    if Fourier:
      jac = tf.transpose(jac, [0,2,1])
    else:
      jac = tf.linalg.inv(jac)
    jac = tf.pad(jac, tf.constant([[0, 0], [0, 1],[0,1]]) )
    jac = jac + tf.pad(tf.reshape(tf.ones_like(g1), [-1,1,1]), tf.constant([[0,0],[2,0],[2,0]]))
    return jac

#def nshear(img, g1, g2):
#  """ Convenience function to apply a shear to an input image or kimage.
#  """
#  transform_matrix = shear_transformation(g1, g2,
#                                          Fourier=img.dtype == tf.complex64)
#  return transform(img, transform_matrix)

def shear(img,g1,g2):
  
  _ , nx, ny, _ = img.get_shape().as_list()
  g1 = tf.convert_to_tensor(g1, dtype=tf.float32)
  g2 = tf.convert_to_tensor(g2, dtype=tf.float32)
  gsqr = g1**2 + g2**2
  
  # Building a batched jacobian
  jac = tf.stack([ 1. + g1, g2,
                g2, 1. - g1], axis=1) / tf.expand_dims(tf.sqrt(1.- gsqr),1)
  jac = tf.reshape(jac, [-1,2,2]) 

  # Inverting these jacobians to follow the TF definition
  if img.dtype == tf.complex64:
    transform_matrix = tf.transpose(jac,[0,2,1])
  else:
    transform_matrix = tf.linalg.inv(jac)
  
  #define a grid at pixel positions
  warp = tf.stack(tf.meshgrid(tf.linspace(0.,tf.cast(nx,tf.float32)-1.,nx), 
                              tf.linspace(0.,tf.cast(ny,tf.float32)-1.,ny)),axis=-1)[..., tf.newaxis]

  #get center
  center = tf.convert_to_tensor([[nx/2],[ny/2]],dtype=tf.float32)
  
  #displace center to origin
  warp = warp - center
  
  #if fourier, no half pixel shift needed
  if  img.dtype is not tf.complex64:
    warp +=.5

  #apply shear
  warp = tf.matmul(transform_matrix[:, tf.newaxis, tf.newaxis, ...], warp)[...,0]

  #return center
  warp = warp + center[...,0] 
 
  #if fourier, no half pixel shift needed
  if  img.dtype is not tf.complex64:
    warp -=.5
    
  #apply resampler
  if img.dtype == tf.complex64:
    a = tfa_image.resampler(tf.math.real(img),warp,'bernsteinquintic')
    b = tfa_image.resampler(tf.math.imag(img),warp,'bernsteinquintic')
    sheared = tf.complex(a,b)
  else:
    sheared = tfa_image.resampler(img,warp,'bernsteinquintic')
  return sheared