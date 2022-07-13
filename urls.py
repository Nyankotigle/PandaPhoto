from django.conf.urls import include,url
from . import views
from django.conf.urls.static import static
from django.conf import settings
app_name = 'panda_shot'

urlpatterns = [
        url(r'^$', views.demo_pose, name='demo_pose'),
        url(r'^take_photo/$', views.take_photo, name='take_photo'),
        url(r'^InformMatchSuccess/$', views.InformMatchSuccess, name='InformMatchSuccess'),
        url(r'^panda_video/$', views.panda_video, name='panda_video'),
        url(r'^WrapGAN/$', views.WrapGAN, name='WrapGAN'),
    ]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)