import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id, row_number
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.window import Window


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Create a spark session for the project to interact with various spark's functionality
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads song data from S3 and transform them into songs and artists table,
    and write them on the sparkify S3
    
    Arguments:
        spark {object}: spark session
        input_data {string}: a filepath to S3 where contains song data
        output_data {string}: a filepath to sparkify S3
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.filter(df.song_id != '') \
        .select("song_id", "title", "artist_id", "year", "duration") \
        .dropDuplicates()
    
    # output filepath to songs table file
    songs_table_path = output_data + "songs_table.parquet"
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").mode("overwrite") \
        .parquet(songs_table_path)

    # extract columns to create artists table
    artists_table = df.filter(df.artist_id != '').select("artist_id",
        "artist_name", "artist_location", "artist_latitude", "artist_longitude") \
        .dropDuplicates()
    
    # output filepath to artists table file
    artists_table_path = output_data + "artists_table.parquet"
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite") \
        .parquet(artists_table_path)


def process_log_data(spark, input_data, output_data):
    """
    Loads log data from S3 and transform them into users, time and songplays table,
    and write them on the sparkify S3
    
    Arguments:
        spark {object}: spark session
        input_data {string}: a filepath to S3 where contains log data
        output_data {string}: a filepath to sparkify S3
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.filter(df.userId != '').selectExpr("userId as user_id",
        "firstName as first_name", "lastName as last_name", "gender", "level") \
        .dropDuplicates()
    
    # output filepath to users table file
    users_table_path = output_data + "users_table.parquet"
    
    # write users table to parquet files
    users_table.write.mode("overwrite") \
        .parquet(users_table_path)

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x:  datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumn("start_time", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
    df = df.withColumn("datetime", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.select("start_time", hour("start_time").alias("hour"), dayofmonth("datetime").alias("day"),
            weekofyear("datetime").alias("week"), month("datetime").alias("month"),
            year("datetime").alias("year"), dayofweek("datetime").alias("weekday")).dropDuplicates()
    
    # output filepath to time table
    time_table_path = output_data + "time_table.parquet"
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").mode("overwrite") \
        .parquet(time_table_path)
    
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"

    # read in song data to use for songplays table
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, (df.song == song_df.title) & (df.length == song_df.duration) & 
                         (df.artist == song_df.artist_name), how='left').dropDuplicates()
    songplays_table = songplays_table.withColumn("id", monotonically_increasing_id())
    windowSpec = Window.orderBy("id")
    songplays_table.withColumn("songplay_id", row_number().over(windowSpec))
    songplays_table = songplays_table.selectExpr("songplay_id", "start_time", 
            "userId as user_id", "level", "song_id", "artist_id", "sessionId as session_id",
            "location", "userAgent as user_agent", "year(start_time) as year",
            "month(start_time) as month")

    # output filepath to songplays table
    songplays_table_path = output_data + "songplays_table.parquet"
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").mode("overwrite") \
        .parquet(songplays_table_path)

    
def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://sparkify-database/"
    
    # Load song and log data, transform them into the appropriate tables and write 
    # them into S3
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
