from __future__ import annotations

from datetime import datetime, timezone, timedelta

from utils.env_util.runtime_env import get_timezone_name


def get_timezone() -> timezone:
    """从环境变量获取时区设置。

    环境变量: NANOAGENT_TIMEZONE
    默认值: shanghai (UTC+8)

    Returns:
        timezone对象
    """
    tz_name = get_timezone_name("shanghai").lower().strip()

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


def timezone_label() -> str:
    """获取当前配置时区的人类可读标签。

    Returns:
        形如 UTC+8 / UTC-5 / UTC
    """
    offset = now().utcoffset() or timedelta()
    total_minutes = int(offset.total_seconds() // 60)
    if total_minutes == 0:
        return "UTC"
    sign = "+" if total_minutes >= 0 else "-"
    abs_minutes = abs(total_minutes)
    hours, minutes = divmod(abs_minutes, 60)
    if minutes == 0:
        return f"UTC{sign}{hours}"
    return f"UTC{sign}{hours}:{minutes:02d}"


def log_timestamp() -> str:
    """获取日志时间前缀字符串。

    Returns:
        形如 2026-04-08 12:34:56:789 UTC+8
    """
    current = now()
    milliseconds = current.microsecond // 1000
    return f"{current.strftime('%Y-%m-%d %H:%M:%S')}:{milliseconds:03d} {timezone_label()}"
