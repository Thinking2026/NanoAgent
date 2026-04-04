from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta


def get_timezone() -> timezone:
    """从环境变量获取时区设置。

    环境变量: NANOAGENT_TIMEZONE
    默认值: shanghai (UTC+8)

    Returns:
        timezone对象
    """
    tz_name = os.environ.get("NANOAGENT_TIMEZONE", "shanghai").lower().strip()

    tz_map = {
        "shanghai": timezone(timedelta(hours=8)),
        "beijing": timezone(timedelta(hours=8)),  # 北京时间
        "utc": timezone.utc,
        "gmt": timezone.utc,
    }

    return tz_map.get(tz_name, timezone(timedelta(hours=8)))


def now() -> datetime:
    """获取当前时区的datetime对象。

    Returns:
        当前时区的datetime对象
    """
    return datetime.now(get_timezone())


def isoformat(timespec: str = "seconds") -> str:
    """获取ISO格式的当前时间字符串。

    Args:
        timespec: 时间精度 ('auto', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds')

    Returns:
        ISO格式的时间字符串
    """
    return now().isoformat(timespec=timespec)


def strftime(format_str: str) -> str:
    """使用指定格式获取当前时间字符串。

    Args:
        format_str: 时间格式字符串，如 "%Y-%m-%d %H:%M:%S"

    Returns:
        格式化的时间字符串
    """
    return now().strftime(format_str)


def timestamp_full() -> str:
    """获取完整的日期时间戳 (YYYY-MM-DD HH:MM:SS)。

    Returns:
        格式为 YYYY-MM-DD HH:MM:SS 的时间戳字符串
    """
    return strftime("%Y-%m-%d %H:%M:%S")


def timestamp_date() -> str:
    """获取日期字符串 (YYYY-MM-DD)。

    Returns:
        格式为 YYYY-MM-DD 的日期字符串
    """
    return strftime("%Y-%m-%d")


def utc_now() -> datetime:
    """获取UTC时区的当前时间。

    Returns:
        UTC时区的datetime对象
    """
    return datetime.now(timezone.utc)
    """获取UTC时区的当前时间。

    Returns:
        UTC时区的datetime对象
    """
    return datetime.now(timezone.utc)


def utc_isoformat(timespec: str = "seconds") -> str:
    """获取UTC时区的ISO格式时间字符串。

    Args:
        timespec: 时间精度 ('auto', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds')

    Returns:
        UTC时区的ISO格式时间字符串
    """
    return utc_now().isoformat(timespec=timespec)