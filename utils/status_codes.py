class StatusCodes:
    # Task stage
    SUCCESS         = "AC_000"
    PENDING         = "AC_001" 
    INPROGRESS      = "AC_002"
    
    # Client
    INVALID_REQUEST                 =  "AC_400" # Empty, null, invalid filed in payload
    EXCEEDING_PERMITTED_RESOURCES   = "AC_401"  # < 300s is permitted
    RESOURCE_DOES_NOT_EXIST         = "AC_402"  # Can not find melody for exmaple
    UNSUPPORTED                     = "AC_403"  # Type of resource: melody must be *mp3

    # Server
    TIMEOUT         = "AC_500" # If a task exceeding timeout => Set status timeout
    ERROR           = "AC_501" # unknown ERROR
    RABBIT_ERROR    = "AC_502" # Service cannot connect to Rabbit
    REDIS_ERROR     = "AC_303" # Service cannot connect to Redis
    S3_ERROR        = "AC_504" # Service cannot connect to S3