#!/usr/bin/env python
# coding: utf-8

import logging
import traceback


class ErrorHandler:
    @staticmethod
    def log_and_raise(error: Exception, custom_message: str = '') -> None:
        """
        Логирует ошибку и выбрасывает её снова.

        :param error: Исключение (Exception), которое нужно обработать.
        :param custom_message: Пользовательское сообщение для логирования.
        """
        error_message = f"{custom_message}\n{traceback.format_exc()}"
        logging.error(error_message)
        raise error

    @staticmethod
    def log_warning(warning_message: str) -> None:
        """
        Логирует предупреждение.

        :param warning_message: Сообщение о предупреждении.
        """
        logging.warning(warning_message)

    # @staticmethod
    # def log_info(info_message: str) -> None:
    #     """
    #     Логирует информационное сообщение.

    #     :param info_message: Информационное сообщение.
    #     """
    #     logging.info(info_message)