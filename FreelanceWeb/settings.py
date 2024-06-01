from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()



BASE_DIR = Path(__file__).resolve().parent.parent
APPONE_TEMPLATE_DIR = Path.joinpath(BASE_DIR,'onepapge/templates')
APPONE_STATIC_DIR = Path.joinpath(BASE_DIR, 'onepage/static')
# capstone = Path.joinpath(BASE_DIR,'Capstone')

SECRET_KEY = 'django-insecure-(7)ub=7g6_rw!l6e+jgmenz3yhy(m#w$if*6+z)n+c*qd2fn1)'
DEBUG = False

ALLOWED_HOSTS = ['*']



INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'onepage',
]

MIDDLEWARE = [
     'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'FreelanceWeb.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [APPONE_TEMPLATE_DIR],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'FreelanceWeb.wsgi.application'
CORS_ALLOW_CREDENTIALS = True

CORS_ORIGIN_ALLOW_ALL = True

CORS_ALLOW_CREDENTIALS = True


CSRF_TRUSTED_ORIGINS = ['https://www.cruzai.tech', 'https://cruzai.tech']

CORS_REPLACE_HTTPS_REFERER = True

CSRF_COOKIE_DOMAIN = 'cruzai.tech.com'

CORS_ORIGIN_WHITELIST = (
    'https://cruzai.tech',
    'https://www.cruzai.tech',
)



# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/


import os
# if DEBUG:
STATICFILES_DIRS = [
        os.path.join(BASE_DIR, 'onepage/static')
    ]
# else:
STATIC_URL = '/onepage/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'onepage/staticfiles')



# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
