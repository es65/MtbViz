"""
manual paste from mtb2 repo
"""

# Export all models
__all__ = [
    "User",
    "Bike",
    "Ride",
    "Jump",
    "JumpGlobal",
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

from .session import Base


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

    # Relationships
    rides = relationship("Ride", back_populates="user")
    bikes = relationship("Bike", back_populates="user", cascade="all, delete-orphan")


class Bike(Base):
    """Model for user's bikes/bicycles."""

    __tablename__ = "bikes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name = Column(String(100), nullable=False)  # nickname/display name
    brand = Column(String(100))  # manufacturer (e.g., "Trek", "Specialized")
    model = Column(String(100))  # specific model
    bike_type = Column(String(50))  # mountain, road, gravel, hybrid, etc.
    wheel_size = Column(String(20))  # e.g., "29", "27.5", "26", "700c"
    year = Column(Integer)  # model year
    weight = Column(Double)  # bike weight in kg
    notes = Column(String(500))  # user notes/custom details
    is_active = Column(Boolean, default=True)  # soft delete flag
    is_default = Column(Boolean, default=False)  # default bike for rides
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at = Column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    user = relationship("User", back_populates="bikes")
    rides = relationship("Ride", back_populates="bike")

    # Indexes for performance
    __table_args__ = (
        Index("idx_bikes_user_id", "user_id"),
        Index("idx_bikes_user_active", "user_id", "is_active"),
    )


class Ride(Base):
    __tablename__ = "rides"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=False
    )
    bike_id = Column(
        UUID(as_uuid=True), ForeignKey("bikes.id", ondelete="SET NULL"), nullable=True
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
    bike = relationship("Bike", back_populates="rides")
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
    jump_global_id = Column(
        UUID(as_uuid=True),
        ForeignKey("jumps_global.id", ondelete="SET NULL"),
        nullable=True,
    )  # Link to global jump entity (NULL if classified as noise)
    ts = Column(TIMESTAMP(timezone=True, precision=6))  # timestamp of jump event
    location = Column(
        JSON
    )  # {lat: float, lng: float}; deprecated but keep until manual migration
    distance = Column(Double)
    airtime = Column(Double)
    lat_takeoff = Column(Double)
    lng_takeoff = Column(Double)
    lat_landing = Column(Double)
    lng_landing = Column(Double)
    boost = Column(Double)  # integrated pre-jump vertical acceleration (m/s)
    impact = Column(Double)  # integrated post-landing vertical acceleration (m/s)
    speed_mph = Column(Double)  # speed during jump
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at = Column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    ride = relationship("Ride", back_populates="jumps")
    jump_global = relationship("JumpGlobal", back_populates="jumps")

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_jumps_ride_ts", "ride_id", "ts"),
        # Spatial indexes for clustering queries
        Index("idx_jumps_takeoff_coords", "lat_takeoff", "lng_takeoff"),
        Index("idx_jumps_landing_coords", "lat_landing", "lng_landing"),
    )


class JumpGlobal(Base):
    """Global jump entities derived from clustering individual jump events."""

    __tablename__ = "jumps_global"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at = Column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    # Takeoff cluster statistics
    lat_takeoff_avg = Column(Double, nullable=False)
    lng_takeoff_avg = Column(Double, nullable=False)
    lat_takeoff_std = Column(Double, nullable=False)
    lng_takeoff_std = Column(Double, nullable=False)

    # Landing cluster statistics
    lat_landing_avg = Column(Double, nullable=False)
    lng_landing_avg = Column(Double, nullable=False)
    lat_landing_std = Column(Double, nullable=False)
    lng_landing_std = Column(Double, nullable=False)

    # Jump metrics: averages
    distance_avg = Column(Double)
    airtime_avg = Column(Double)
    boost_avg = Column(Double)
    impact_avg = Column(Double)

    # Jump metrics: standard deviations
    distance_std = Column(Double)
    airtime_std = Column(Double)
    boost_std = Column(Double)
    impact_std = Column(Double)

    # Metadata
    jump_count = Column(Integer, nullable=False)  # number of jumps in this cluster

    # Relationship: back-reference to individual jumps in this cluster
    jumps = relationship("Jump", back_populates="jump_global")

    # Indexes for spatial queries
    __table_args__ = (
        Index("idx_jumps_global_takeoff_lat", "lat_takeoff_avg"),
        Index("idx_jumps_global_takeoff_lng", "lng_takeoff_avg"),
        Index("idx_jumps_global_landing_lat", "lat_landing_avg"),
        Index("idx_jumps_global_landing_lng", "lng_landing_avg"),
        # Composite indexes for bounding box queries
        Index("idx_jumps_global_takeoff_coords", "lat_takeoff_avg", "lng_takeoff_avg"),
        Index("idx_jumps_global_landing_coords", "lat_landing_avg", "lng_landing_avg"),
    )


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
