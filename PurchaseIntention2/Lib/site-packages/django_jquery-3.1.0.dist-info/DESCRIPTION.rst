Django jQuery
=============


Requirements
------------

`Django <https://www.djangoproject.com/>`_ 1.3 or later


Installation
------------

::

    $ pip install django-jquery


Setup
-----

Just add ``'django.contrib.staticfiles'`` and ``'jquery'`` to INSTALLED_APPS in
your settings.py::

    INSTALLED_APPS = (
        # ...

        'django.contrib.staticfiles',
        'jquery',

        # ...
    )

Refer to Django `static files <https://docs.djangoproject.com/en/dev/howto/static-files/>`_
documentation to configure and deploy static files.


Usage
-----

You can refer to jquery in your template with::

    {{ STATIC_URL }}js/jquery.js


Admin template customization::

    {% extends "admin/base_site.html" %}

    {% block extrahead %}
        <script type="text/javascript" src="{{ STATIC_URL }}js/jquery.js" />
    {% endblock %}


Custom widget::

    class MyWidget(forms.TextInput):
        class Media:
            js = ('js/jquery.js',)

        def render(self, name, value, attrs=None):
            html = super(MyWidget, self).render(name, value, attrs=attrs)
            # ...
            return html


