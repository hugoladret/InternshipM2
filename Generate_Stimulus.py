#!/usr/bin/env python
# coding: utf-8

# # Notebook de générations de MotionClouds
# ## Pour Nelson Cortés Hernández, du labo de Christian Casanova

# Les imports classiques

# In[1]:


import numpy as np
import MotionClouds as mc
import imageio
import matplotlib.pyplot as plt


# Les paramètres de sauvegarde

# On défini la résolution en x-y (pixels) et t (ms)

# In[2]:


fx, fy, ft = mc.get_grids(512, 512, 128)


# L'envelope gabor du stimulus est généré, avec les paramètres :
# * theta l'angle du stimulus
# * b_theta l'ouverture des gabors (donc le niveau de bruit)
# * sf_0 et b_sf les fréquences spatiales

# In[3]:


envelope = mc.envelope_gabor(fx, fy, ft,
                             V_X=1., V_Y=0., B_V=.1,
                             sf_0=.15, B_sf=.1,
                             theta=np.pi/3, B_theta=np.pi/12, alpha=1.)


# On génère le cloud

# In[4]:


movie = mc.random_cloud(envelope)
movie = mc.rectif(movie)


# Une image instantanée à t = 0

# In[5]:


plt.figure(figsize = (8,6))
plt.imshow(movie[:,:,0], cmap = plt.cm.gray)


# Et on sauvegarde le stimulus au format desiré (paramètre vext)
# * Pour faire du .mp4, il faut installer ffmpeg (sudo apt install ffmpeg), c'est la faute a Linux, désolé !
# * Pour faire du .png, le chemin spécifié devient un fichier dans lequel seront organisé les images

# In[15]:


mc.anim_save(movie, './stimulus', display=False, vext='.mp4', verbose = True)


# Pour afficher le code

# In[11]:


get_ipython().run_line_magic('pinfo2', 'mc.anim_save')


# Et en bonus, si jamais ça marche chez vous, vous pouvez montrer le stimulus directement dans le notebook, mais on a quelques soucis de compatibilité avec les lecteurs Flash sur Ubuntu

# In[ ]:


for sf_0 in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    name_ = name + '-sf_0-' + str(sf_0).replace('.', '_')
    # function performing plots for a given set of parameters
    mc.figures_MC(fx, fy, ft, name_, sf_0=sf_0, **opts)
    mc.in_show_video(name_, **opts)

