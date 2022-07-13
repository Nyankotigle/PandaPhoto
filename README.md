<h1>PandaPhoto</h1>
<h2>熊猫互动拍照系统</h2>
——熊猫主题文娱创作平台的子系统，旨在探索先进的计算机视觉算法在大熊猫主题的互动拍照场景上的应用。
<br />
<br />
系统结合人脸及人体关键点识别，人像分割，目标检测，图像风格迁移，以及自己设计实现的熊猫分割PandaSeg，动作识别PoseRecognition等算法，依托Django框架搭建的Web应用，在服务器端使用 tensorflow、pytorch等深度学习框架搭建的智能图像处理模块处理前端通过单目相机捕获的图片并实时返回处理结果，目前可以实现实时视频挂件，人脸表情包生成，人像与熊猫照片创意融合，多动作互动拍照，分区风格化等功能。
<h2>系统结构</h2>
<div align=center><img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E7%B3%BB%E7%BB%9F%E6%9E%B6%E6%9E%84.png" width="90%" height="90%"></div>
<br />
<h2>系统注册、登录界面：</h2>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/register.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/login.png" width="45%" height="45%">
</div>
<br />
<h2>系统主界面：</h2>
<div align=center><img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E4%B8%BB%E9%A1%B5%E9%9D%A2.png" width="90%" height="90%"></div>
<br />
<h2>熊猫贴纸拍照模块：</h2>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E7%86%8A%E7%8C%AB%E8%B4%B4%E7%BA%B8.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E7%86%8A%E7%8C%AB%E8%B4%B4%E7%BA%B8%E6%95%88%E6%9E%9C.png" width="45%" height="45%">
</div>
<br />
<h2>熊猫头表情包拍照模块：</h2>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E7%86%8A%E7%8C%AB%E8%A1%A8%E6%83%85.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E7%86%8A%E7%8C%AB%E8%A1%A8%E6%83%852.png" width="45%" height="45%">
</div>
<br />
<h2>熊猫背景环境创意融合模块：</h2>
<div align=center><img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E7%86%8A%E7%8C%AB%E7%8E%AF%E5%A2%83%E8%9E%8D%E5%90%88%E9%A1%B5%E9%9D%A2.png" width="90%" height="90%"></div>
<br />
<h3>动作识别拍照（互动融合）：</h3>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E5%8A%A8%E4%BD%9C%E5%B1%95%E7%A4%BA.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E5%8A%A8%E4%BD%9C%E8%AF%86%E5%88%AB%E6%8B%8D%E7%85%A7%E7%BB%93%E6%9E%9C%E5%B1%95%E7%A4%BA.png" width="45%" height="45%">
</div>
<br />
<h3>定时拍照（自动融合）：</h3>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E5%AE%9A%E6%97%B6%E6%8B%8D%E7%85%A7%E5%80%92%E8%AE%A1%E6%97%B6.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E5%AE%9A%E6%97%B6%E6%8B%8D%E7%85%A7%E7%BB%93%E6%9E%9C%E5%B1%95%E7%A4%BA.png" width="45%" height="45%">
</div>
<br />
<h3>视频背景替换：</h3>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E8%A7%86%E9%A2%91%E8%9E%8D%E5%90%88.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E8%A7%86%E9%A2%91%E8%9E%8D%E5%90%88%E5%A4%84%E7%90%86%E4%B8%AD.png" width="45%" height="45%">
</div>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E8%A7%86%E9%A2%91%E8%9E%8D%E5%90%88%E5%BC%80%E5%A7%8B%E5%BD%95%E5%83%8F.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E8%A7%86%E9%A2%91%E8%9E%8D%E5%90%88%E7%BB%93%E6%9E%9C%E5%B1%95%E7%A4%BA.png" width="45%" height="45%">
</div>
<br />
<h3>风格化处理：</h3>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E9%A3%8E%E6%A0%BC%E5%8C%96%E9%A1%B5%E9%9D%A2.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E9%A3%8E%E6%A0%BC%E5%8C%96%E9%80%89%E6%8B%A9.png" width="45%" height="45%">
</div>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E9%A3%8E%E6%A0%BC%E5%8C%96%E7%BB%93%E6%9E%9C.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E9%A3%8E%E6%A0%BC%E5%8C%96%E7%BB%93%E6%9E%9C2.png" width="45%" height="45%">
</div>
<br />
<h2>动漫头像生成模块：</h2>
<div align=center>
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E5%8A%A8%E6%BC%AB%E5%A4%B4%E5%83%8F.png" width="45%" height="45%">
<img src="https://github.com/Nyankotigle/PandaPhoto/blob/master/images/%E5%8A%A8%E6%BC%AB%E5%A4%B4%E5%83%8F2.png" width="45%" height="45%">
</div>
<br />
