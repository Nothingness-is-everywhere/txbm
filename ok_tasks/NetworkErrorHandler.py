from ok import TriggerTask


class NetworkErrorHandler(TriggerTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "网络错误处理"
        self.description = "检测并关闭网络错误弹窗（如'网络不好'等提示）"
        self.trigger_interval = 2
        self.visible = True

    def run(self):
        network_error_texts = ["网络不好", "网络异常", "连接失败", "网络连接", "无法连接", "确认"]
        for text in network_error_texts:
            boxes = self.ocr(match=text)
            if boxes:
                self.logger.info(f"检测到网络错误提示: '{text}'")
                self.click_box(boxes[0])
                return True
        return False