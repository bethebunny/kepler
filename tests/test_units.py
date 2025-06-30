import pytest
from kepler.reporting.units import MetricPrefix, Bytes, Time


class TestUnit:
    """Test the base Unit class functionality."""

    def test_unit_ordering(self):
        """Test that units can be ordered by their order value."""
        # Using Bytes for concrete testing
        assert Bytes.B < Bytes.KiB
        assert Bytes.KiB < Bytes.MiB
        assert Bytes.GiB > Bytes.MiB

    def test_unit_comparison_type_safety(self):
        """Test that comparing different unit types raises TypeError."""
        with pytest.raises(TypeError):
            Bytes.B < Time.SECOND

    def test_unit_format_basic(self):
        """Test basic formatting functionality."""
        # Test with Bytes - improved formatting avoids scientific notation for readable numbers
        result = Bytes.B._format(1024, precision=3)
        assert result == "1024 B"  # Much better than "1.02e+03 B"

        result = Bytes.KiB._format(1024, precision=3)
        assert result == "1 KiB"

    def test_unit_format_precision(self):
        """Test formatting with different precision values."""
        result = Bytes.B._format(1234, precision=2)
        assert result == "1.2e+03 B"

        result = Bytes.B._format(1234, precision=4)
        assert result == "1234 B"

    def test_best_unit_selection(self):
        """Test best unit selection logic."""
        # Should choose KiB for 1024 bytes
        assert Bytes.best_unit(1024) == Bytes.KiB

        # Should choose MiB for 1MB worth of bytes
        assert Bytes.best_unit(1024 * 1024) == Bytes.MiB

        # For very large values that exceed the 1-1000 range for largest unit,
        # it defaults to the base unit (B)
        assert Bytes.best_unit(1024**10) == Bytes.B

        # Should default to smallest unit for very small values
        assert Bytes.best_unit(0.5) == Bytes.B

    def test_default_unit(self):
        """Test that default returns the smallest unit."""
        assert Bytes.default() == Bytes.B
        assert Time.default() == Time.SECOND

    def test_format_class_method(self):
        """Test the class-level format method."""
        result = Bytes.format(1024)
        assert result == "1 KiB"

        result = Bytes.format(1024 * 1024 * 1.5)
        assert result == "1.5 MiB"


class TestMetricPrefix:
    """Test MetricPrefix unit functionality."""

    def test_metric_prefix_values(self):
        """Test that metric prefix values are correct."""
        assert MetricPrefix.QUECTO.order == 1e-30
        assert MetricPrefix.RONTO.order == 1e-27
        assert MetricPrefix.YOCTO.order == 1e-24
        assert MetricPrefix.ZEPTO.order == 1e-21
        assert MetricPrefix.ATTO.order == 1e-18
        assert MetricPrefix.FEMTO.order == 1e-15
        assert MetricPrefix.PICO.order == 1e-12
        assert MetricPrefix.NANO.order == 1e-9
        assert MetricPrefix.MICRO.order == 1e-6
        assert MetricPrefix.MILLI.order == 1e-3
        assert MetricPrefix.UNIT.order == 1
        assert MetricPrefix.KILO.order == 1e3
        assert MetricPrefix.MEGA.order == 1e6
        assert MetricPrefix.GIGA.order == 1e9
        assert MetricPrefix.TERA.order == 1e12
        assert MetricPrefix.PETA.order == 1e15
        assert MetricPrefix.EXA.order == 1e18
        assert MetricPrefix.ZETA.order == 1e21
        assert MetricPrefix.YOTTA.order == 1e24
        assert MetricPrefix.RONNA.order == 1e27
        assert MetricPrefix.QUETTA.order == 1e30

    def test_metric_prefix_shortnames(self):
        """Test that metric prefix shortnames are correct."""
        assert MetricPrefix.NANO.shortname == "n"
        assert MetricPrefix.MICRO.shortname == "μ"
        assert MetricPrefix.MILLI.shortname == "m"
        assert MetricPrefix.UNIT.shortname == ""
        assert MetricPrefix.KILO.shortname == "k"
        assert MetricPrefix.MEGA.shortname == "M"
        assert MetricPrefix.GIGA.shortname == "G"

    def test_metric_prefix_default(self):
        """Test that default returns UNIT."""
        assert MetricPrefix.default() == MetricPrefix.UNIT

    def test_format_unit(self):
        """Test the format_unit method."""
        # Test with seconds
        result = MetricPrefix.format_unit("s", 0.001)
        assert result == "1 ms"

        result = MetricPrefix.format_unit("s", 1000)
        assert result == "1 ks"

        # Test with custom unit
        result = MetricPrefix.format_unit("Hz", 1000000)
        assert result == "1 MHz"


class TestBytes:
    """Test Bytes unit functionality."""

    def test_byte_unit_values(self):
        """Test that byte unit values are correct powers of 2."""
        assert Bytes.B.order == 1
        assert Bytes.KiB.order == 2**10
        assert Bytes.MiB.order == 2**20
        assert Bytes.GiB.order == 2**30
        assert Bytes.TiB.order == 2**40
        assert Bytes.PiB.order == 2**50
        assert Bytes.EiB.order == 2**60
        assert Bytes.ZiB.order == 2**70
        assert Bytes.YiB.order == 2**80

    def test_byte_unit_shortnames(self):
        """Test that byte unit shortnames are correct."""
        assert Bytes.B.shortname == "B"
        assert Bytes.KiB.shortname == "KiB"
        assert Bytes.MiB.shortname == "MiB"
        assert Bytes.GiB.shortname == "GiB"

    def test_byte_formatting(self):
        """Test byte formatting for common values."""
        assert Bytes.format(512) == "512 B"
        assert Bytes.format(1024) == "1 KiB"
        assert Bytes.format(1536) == "1.5 KiB"
        assert Bytes.format(1024 * 1024) == "1 MiB"
        assert Bytes.format(1024 * 1024 * 1024) == "1 GiB"

    def test_byte_formatting_fractional(self):
        """Test that fractional bytes are handled correctly."""
        result = Bytes.format(1024.5)
        # Should be close to 1 KiB
        assert "1" in result and "KiB" in result

    def test_improved_byte_formatting_exact(self):
        """Test the improved byte formatting that avoids unnecessary scientific notation."""
        # The improved formatting should show human-readable numbers instead of scientific notation
        assert Bytes.format(1000) == "1000 B"  # Was "1e+03 B"
        assert Bytes.format(1023) == "1023 B"  # Was "1.02e+03 B"
        assert Bytes.format(1000.5) == "1000 B"  # Fractional values work too
        assert Bytes.format(9999) == "9.76 KiB"  # Still switches units appropriately

        # But still uses scientific notation for very large/small values
        huge_value = 1024**10
        result = Bytes.format(huge_value)
        assert "e+" in result  # Should still use scientific notation for huge values

        # And for very small values
        assert Bytes.format(0.00001) == "1e-05 B"


class TestTime:
    """Test Time unit functionality."""

    def test_time_unit_values(self):
        """Test that time unit values are correct."""
        assert Time.SECOND.order == 1
        assert Time.MINUTE.order == 60
        assert Time.HOUR.order == 60 * 60
        assert Time.DAY.order == 24 * 60 * 60
        assert Time.YEAR.order == 365 * 24 * 60 * 60

    def test_time_unit_shortnames(self):
        """Test that time unit shortnames are correct."""
        assert Time.SECOND.shortname == "s"
        assert Time.MINUTE.shortname == "m"
        assert Time.HOUR.shortname == "h"
        assert Time.DAY.shortname == "d"
        assert Time.YEAR.shortname == "y"

    def test_format_nanos(self):
        """Test format_nanos method."""
        # 1 second in nanoseconds
        result = Time.format_nanos(1e9)
        assert result == "1 s"

        # 1 millisecond in nanoseconds
        result = Time.format_nanos(1e6)
        assert result == "1 ms"

        # 1 microsecond in nanoseconds
        result = Time.format_nanos(1e3)
        assert result == "1 μs"

        # 1 nanosecond
        result = Time.format_nanos(1)
        assert result == "1 ns"

    def test_time_format_sub_second(self):
        """Test formatting for sub-second time values uses SI prefixes."""
        assert Time.format(0.001) == "1 ms"
        assert Time.format(0.000001) == "1 μs"
        assert Time.format(0.000000001) == "1 ns"

    def test_time_format_seconds_and_above(self):
        """Test formatting for seconds and larger time units."""
        # Test seconds
        assert Time.format(1) == "1 s"
        assert Time.format(30) == "30 s"

        # Test minutes (should use compound format)
        result = Time.format(90)  # 1m30s
        assert "1m" in result and "30s" in result

        # Test hours
        result = Time.format(3661)  # 1h1m
        assert "1h" in result and "1m" in result

        # Test days
        result = Time.format(90000)  # 1d1h
        assert "1d" in result and "1h" in result

        # Test years
        result = Time.format(365 * 24 * 60 * 60 + 24 * 60 * 60)  # 1y1d
        assert "1y" in result and "1d" in result

    def test_time_format_compound_zero_subunits(self):
        """Test that compound formatting handles zero subunits correctly."""
        # Exactly 1 minute should show 1m0s
        result = Time.format(60)
        assert "1m0s" in result

        # Exactly 1 hour should show 1h0m
        result = Time.format(3600)
        assert "1h0m" in result


class TestUnitIntegration:
    """Integration tests across unit types."""

    def test_best_unit_boundary_conditions(self):
        """Test best_unit selection at boundary conditions."""
        # Test exactly at boundaries
        assert Bytes.best_unit(1000) == Bytes.B  # Just under 1 KiB
        assert Bytes.best_unit(1024) == Bytes.KiB  # Exactly 1 KiB

        # Test very large values - 1024**8 is exactly 1 YiB
        huge_value = 1024**8
        unit = Bytes.best_unit(huge_value)
        assert unit == Bytes.YiB  # Should return YiB for exactly 1 YiB

        # Test value that exceeds the range - should default to base unit
        super_huge_value = 1024**10
        unit = Bytes.best_unit(super_huge_value)
        assert unit == Bytes.B  # Should default to base unit for values exceeding range

    def test_precision_edge_cases(self):
        """Test formatting with edge case precision values."""
        # When formatting bytes, it chooses the best unit first
        result = Bytes.format(1234.5678, precision=6)
        assert "KiB" in result  # Should be formatted in KiB, not raw bytes

        # Precision of 1
        result = Bytes.format(1234, precision=1)
        assert "KiB" in result  # Should be formatted in KiB

    def test_zero_and_negative_values(self):
        """Test handling of zero and negative values."""
        # Zero should use default unit
        result = Bytes.format(0)
        assert result == "0 B"

        # Negative values (though unusual, should be handled)
        result = Time.format(-1)
        assert "-1 s" in result


class TestExactBehaviors:
    """Test exact behaviors with string equality for edge cases."""

    def test_very_small_values_exact(self):
        """Test exact formatting of very small values."""
        # Very small bytes - should fall back to base unit
        assert Bytes.format(0.001) == "0.001 B"
        assert Bytes.format(0.0001) == "0.0001 B"
        assert Bytes.format(1e-10) == "1e-10 B"

        # Very small time values - should use SI prefixes
        assert Time.format(1e-12) == "1 ps"
        assert Time.format(1e-15) == "1 fs"
        assert Time.format(1e-18) == "1 as"
        assert Time.format(1e-21) == "1 zs"
        assert Time.format(1e-24) == "1 ys"

        # Extremely small time values - use exotic SI prefixes
        assert Time.format(1e-27) == "1 rs"
        assert Time.format(1e-30) == "1 qs"

    def test_very_large_values_exact(self):
        """Test exact formatting of very large values that exceed normal ranges."""
        # Very large byte values - fall back to base unit with scientific notation
        huge_bytes = 1024**10  # Way beyond YiB range
        assert Bytes.format(huge_bytes) == "1.27e+30 B"

        # Even larger values
        gigantic_bytes = 1024**15
        result = Bytes.format(gigantic_bytes)
        assert result.startswith("1.") and result.endswith("e+45 B")

        # Very large time values - when they exceed year range, fall back to SI units!
        huge_time = 1000 * 365 * 24 * 60 * 60  # 1000 years = 31.5 Gs
        assert Time.format(huge_time) == "31.5 Gs"  # Uses SI prefix, not compound!

        # Moderate large time - fits in year range, uses compound format
        gigantic_time = 1e10  # ~317 years
        assert Time.format(gigantic_time) == "317y35d"

    def test_non_round_values_exact(self):
        """Test exact formatting of non-round values."""
        # Non-round byte values - now with correct precision
        assert Bytes.format(1234.56) == "1.21 KiB"  # Correct precision!
        assert Bytes.format(5432.1) == "5.3 KiB"  # Correct precision!
        assert Bytes.format(987654.321) == "965 KiB"  # Correct precision!

        # Non-round time values - sub-second
        assert Time.format(0.00123) == "1.23 ms"
        assert Time.format(0.000000456) == "456 ns"
        assert Time.format(0.000000000789) == "789 ps"

        # Non-round time values - compound format
        assert Time.format(90.5) == "1m30s"  # Truncates fractional seconds
        assert Time.format(3661.7) == "1h1m"  # Truncates fractional minutes
        assert Time.format(93784.2) == "1d2h"  # Truncates fractional hours

    def test_boundary_values_exact(self):
        """Test exact behavior at unit selection boundaries."""
        # Right at the 1-1000 boundary for bytes - now with improved formatting!
        assert Bytes.format(999) == "999 B"
        assert Bytes.format(1000) == "1000 B"  # Much better! No scientific notation
        assert Bytes.format(1023) == "1023 B"  # Much better! No scientific notation
        assert Bytes.format(1024) == "1 KiB"  # Exactly 1 KiB
        assert Bytes.format(1025) == "1 KiB"  # Rounds to 1 KiB with precision=3

        # Upper boundary for KiB - now correctly uses 1024 threshold
        assert Bytes.format(1024 * 999) == "999 KiB"
        assert Bytes.format(1024 * 1000) == "1000 KiB"  # Now correctly shows KiB!
        assert Bytes.format(1024 * 1023) == "1023 KiB"  # Just under threshold
        assert Bytes.format(1024 * 1024) == "1 MiB"  # Exactly 1 MiB

        # Time boundaries
        assert Time.format(0.999) == "999 ms"
        assert Time.format(0.9999) == "1e+03 ms"  # Scientific notation for ms
        assert Time.format(59) == "59 s"
        assert Time.format(59.9) == "59.9 s"
        assert Time.format(60) == "1m0s"  # Switches to compound format

    def test_precision_effects_exact(self):
        """Test how precision affects exact output."""
        # Different precisions for the same value - now with correct precision
        value = 1234567
        assert Bytes.format(value, precision=1) == "1 MiB"  # 1 significant digit
        assert Bytes.format(value, precision=2) == "1.2 MiB"  # 2 significant digits
        assert Bytes.format(value, precision=3) == "1.18 MiB"  # 3 significant digits
        assert Bytes.format(value, precision=4) == "1.177 MiB"  # 4 significant digits

        # With fractional value - same unit selection
        value = 1234567.89
        assert Bytes.format(value, precision=1) == "1 MiB"
        assert Bytes.format(value, precision=2) == "1.2 MiB"
        assert Bytes.format(value, precision=3) == "1.18 MiB"
        assert Bytes.format(value, precision=4) == "1.177 MiB"

    def test_scientific_notation_thresholds_exact(self):
        """Test exact thresholds where scientific notation kicks in."""
        # Values that DON'T trigger scientific notation with improved formatting
        assert Bytes.B._format(100, precision=3) == "100 B"
        assert (
            Bytes.B._format(1000, precision=3) == "1000 B"
        )  # Now shows readable format!
        assert Bytes.B._format(999, precision=3) == "999 B"
        assert Bytes.B._format(999.9, precision=3) == "999.9 B"  # Also readable now!

        # Small values that still don't need scientific notation
        assert Bytes.B._format(0.01, precision=3) == "0.01 B"
        assert Bytes.B._format(0.001, precision=3) == "0.001 B"
        assert Bytes.B._format(0.0001, precision=3) == "0.0001 B"
        assert (
            Bytes.B._format(0.00001, precision=3) == "1e-05 B"
        )  # This one still needs it

    def test_byte_1024_threshold_exact(self):
        """Test that bytes use 1024 threshold instead of 1000."""
        # The key improvement: bytes should use 1024 threshold for unit selection
        assert Bytes.format(1024 * 1000) == "1000 KiB"  # Fixed!
        assert Bytes.format(1024 * 1023) == "1023 KiB"  # Still KiB
        assert Bytes.format(1024 * 1024) == "1 MiB"  # Switches to MiB at 1024 KiB

        # Test the boundary behavior with exact precision
        assert Bytes.format(1024 * 1024 - 1) == "1024 KiB"  # Just under 1 MiB (exact)
        assert Bytes.format(1024 * 1024 + 1) == "1 MiB"  # Just over 1 MiB (rounds)
