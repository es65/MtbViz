"""
Centralized model imports to prevent circular dependencies and multiple table definitions.
"""

# Export all models
__all__ = [
    "User",
    "Ride",
    "Jump",
    "RideMetrics",
    "Location",
    "Compass",
    "Orientation",
    "Barometer",
    "AccelDevice",
    "GyroDevice",
    "SimplifiedTrace",
    "SimplifiedTracePoint",
    "ProcessingQueue",
    "RideChunk",
]

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Boolean,
    PrimaryKeyConstraint,
    Index,
    Double,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from datetime import datetime, timezone
import uuid

from mtb.db.session import Base


def utc_now():
    """Return current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    name = Column(String)
    weight = Column(Double)  # in kg
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    is_active = Column(Boolean, default=True)

    # Relationship: One user can have many rides
    rides = relationship("Ride", back_populates="user")


class Ride(Base):
    __tablename__ = "rides"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=False
    )
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    filename = Column(String)
    file_size = Column(Integer)
    device_info = Column(JSON)
    status = Column(String)  # uploading, uploaded, processing, completed, failed
    title = Column(String(50), nullable=True)  # Optional title, max 50 chars
    description = Column(
        String(500), nullable=True
    )  # Optional description, max 500 chars
    expected_chunks = Column(Integer, default=0)
    chunks_received = Column(Integer, default=0)
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at = Column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    user = relationship("User", back_populates="rides")
    jumps = relationship("Jump", back_populates="ride", cascade="all, delete-orphan")
    metrics = relationship(
        "RideMetrics",
        back_populates="ride",
        uselist=False,
        cascade="all, delete-orphan",
    )
    simplified_traces = relationship(
        "SimplifiedTrace", back_populates="ride", cascade="all, delete-orphan"
    )
    processing_queue = relationship("ProcessingQueue", cascade="all, delete-orphan")
    chunks = relationship("RideChunk", cascade="all, delete-orphan")

    # Timeseries data relationships
    locations = relationship(
        "Location", back_populates="ride", cascade="all, delete-orphan"
    )
    compass_data = relationship(
        "Compass", back_populates="ride", cascade="all, delete-orphan"
    )
    orientations = relationship(
        "Orientation", back_populates="ride", cascade="all, delete-orphan"
    )
    barometer_data = relationship(
        "Barometer", back_populates="ride", cascade="all, delete-orphan"
    )
    accel_data = relationship(
        "AccelDevice", back_populates="ride", cascade="all, delete-orphan"
    )
    gyro_data = relationship(
        "GyroDevice", back_populates="ride", cascade="all, delete-orphan"
    )


class Jump(Base):
    __tablename__ = "jumps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ride_id = Column(UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"))
    location = Column(JSON)  # {lat: float, lng: float}
    distance = Column(Double)
    airtime = Column(Double)
    count = Column(Integer)

    # Relationship: Each jump belongs to one ride
    ride = relationship("Ride", back_populates="jumps")


class RideMetrics(Base):
    """Model for storing processed ride metrics."""

    __tablename__ = "ride_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at = Column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    # Core metrics
    total_duration = Column(Double)
    moving_duration = Column(Double)
    distance = Column(Double)
    average_speed = Column(Double)
    elevation_gain = Column(Double)
    overall_intensity = Column(Double)
    cornering_intensity = Column(Double)
    gyro_intensity = Column(Double)
    longest_jump = Column(Double)
    total_jump_airtime = Column(Double)
    total_jump_distance = Column(Double)

    # Relationship: Each ride has one metrics record
    ride = relationship("Ride", back_populates="metrics")


# Timeseries Tables
class Location(Base):
    __tablename__ = "location"

    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    ts = Column(
        TIMESTAMP(timezone=True, precision=6), nullable=False
    )  # Microsecond precision
    horizontal_accuracy = Column(Double)
    bearing_accuracy = Column(Double)
    altitude = Column(Double)
    longitude = Column(Double)
    vertical_accuracy = Column(Double)
    speed = Column(Double)
    latitude = Column(Double)
    speed_accuracy = Column(Double)
    bearing = Column(Double)

    # Composite primary key
    __table_args__ = (
        PrimaryKeyConstraint("ride_id", "ts"),
        Index("idx_location_ride_ts", "ride_id", "ts"),
    )

    # Relationship: Each location record belongs to one ride
    ride = relationship("Ride", back_populates="locations")


class Compass(Base):
    __tablename__ = "compass"

    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    ts = Column(
        TIMESTAMP(timezone=True, precision=6), nullable=False
    )  # Microsecond precision
    heading = Column(Double)

    # Composite primary key
    __table_args__ = (
        PrimaryKeyConstraint("ride_id", "ts"),
        Index("idx_compass_ride_ts", "ride_id", "ts"),
    )

    # Relationship: Each compass record belongs to one ride
    ride = relationship("Ride", back_populates="compass_data")


class Orientation(Base):
    __tablename__ = "orientation"

    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    ts = Column(
        TIMESTAMP(timezone=True, precision=6), nullable=False
    )  # Microsecond precision
    qx = Column(Double)
    qw = Column(Double)
    qy = Column(Double)
    yaw = Column(Double)
    pitch = Column(Double)
    roll = Column(Double)
    qz = Column(Double)

    # Composite primary key
    __table_args__ = (
        PrimaryKeyConstraint("ride_id", "ts"),
        Index("idx_orientation_ride_ts", "ride_id", "ts"),
    )

    # Relationship: Each orientation record belongs to one ride
    ride = relationship("Ride", back_populates="orientations")


class Barometer(Base):
    __tablename__ = "barometer"

    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    ts = Column(
        TIMESTAMP(timezone=True, precision=6), nullable=False
    )  # Microsecond precision
    pressure = Column(Double)

    # Composite primary key
    __table_args__ = (
        PrimaryKeyConstraint("ride_id", "ts"),
        Index("idx_barometer_ride_ts", "ride_id", "ts"),
    )

    # Relationship: Each barometer record belongs to one ride
    ride = relationship("Ride", back_populates="barometer_data")


class AccelDevice(Base):
    __tablename__ = "accel_device"

    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    ts = Column(
        TIMESTAMP(timezone=True, precision=6), nullable=False
    )  # Microsecond precision
    x = Column(Double)
    y = Column(Double)
    z = Column(Double)

    # Composite primary key
    __table_args__ = (
        PrimaryKeyConstraint("ride_id", "ts"),
        Index("idx_accel_device_ride_ts", "ride_id", "ts"),
    )

    # Relationship: Each accelerometer record belongs to one ride
    ride = relationship("Ride", back_populates="accel_data")


class GyroDevice(Base):
    __tablename__ = "gyro_device"

    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    ts = Column(
        TIMESTAMP(timezone=True, precision=6), nullable=False
    )  # Microsecond precision
    x = Column(Double)
    y = Column(Double)
    z = Column(Double)

    # Composite primary key
    __table_args__ = (
        PrimaryKeyConstraint("ride_id", "ts"),
        Index("idx_gyro_device_ride_ts", "ride_id", "ts"),
    )

    # Relationship: Each gyroscope record belongs to one ride
    ride = relationship("Ride", back_populates="gyro_data")


class SimplifiedTrace(Base):
    __tablename__ = "simplified_traces"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    detail_level = Column(String)  # 'low', 'medium', 'high'
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)

    # Relationships
    ride = relationship("Ride", back_populates="simplified_traces")
    points = relationship(
        "SimplifiedTracePoint",
        back_populates="simplified_trace",
        order_by="SimplifiedTracePoint.sequence_order",
        cascade="all, delete-orphan",
    )


class SimplifiedTracePoint(Base):
    __tablename__ = "simplified_trace_points"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simplified_trace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("simplified_traces.id", ondelete="CASCADE"),
        nullable=False,
    )
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    sequence_order = Column(Integer, nullable=False)  # Preserve order

    # Relationships
    simplified_trace = relationship("SimplifiedTrace", back_populates="points")

    # Composite index for efficient querying
    __table_args__ = (
        Index("idx_trace_sequence", "simplified_trace_id", "sequence_order"),
    )


class ProcessingQueue(Base):
    """Queue for async ride processing."""

    __tablename__ = "processing_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    task_type = Column(
        String(50), default="ride_metrics", nullable=False
    )  # ride_metrics, map_trace
    status = Column(String, default="pending")  # pending, processing, completed, failed
    triggered_by = Column(
        String(50), default="upload", nullable=False
    )  # upload, manual_cli
    priority = Column(Integer, default=0)
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    error_message = Column(String)
    retry_count = Column(Integer, default=0)

    # Index for efficient queue queries
    __table_args__ = (
        Index("idx_processing_queue_status", "status"),
        Index("idx_processing_queue_priority", "priority", "created_at"),
        Index("idx_processing_queue_task_type", "task_type", "status"),
        Index(
            "idx_processing_queue_ride_task_unique", "ride_id", "task_type", unique=True
        ),
    )


class RideChunk(Base):
    """Track uploaded chunks for each ride."""

    __tablename__ = "ride_chunks"

    ride_id = Column(
        UUID(as_uuid=True), ForeignKey("rides.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    uploaded_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    data_points = Column(Integer)

    # Composite primary key
    __table_args__ = (
        PrimaryKeyConstraint("ride_id", "chunk_index"),
        Index("idx_ride_chunks_ride_id", "ride_id"),
    )
