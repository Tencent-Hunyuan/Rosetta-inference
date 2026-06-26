# This script define common functions for other job scripts
#

# ==========================================================================
# Define color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NO_COLOR='\033[0m' # No Color (重置)


# ==========================================================================
# Print colored messages to the terminal for different log levels
# Args:
#   level: The log level (info, warn, error, success), default is info
#   message: The log message to print
print_msg() {
    local level
    case "$1" in
        info|INFO|warn|WARN|error|ERROR|success|SUCCESS)
            level="$1"
            shift
            ;;
        *)
            level="info"
            ;;
    esac

    local message="$*"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    case "$level" in
        info|INFO)
            echo -e "${GREEN}[${timestamp}] [INFO] ${message}${NO_COLOR}"
            ;;
        warn|WARN)
            echo -e "${YELLOW}[${timestamp}] [WARN] ${message}${NO_COLOR}"
            ;;
        error|ERROR)
            echo -e "${RED}[${timestamp}] [ERROR] ${message}${NO_COLOR}"
            ;;
        success|SUCCESS)
            echo -e "${BLUE}[${timestamp}] [SUCCESS] ${message}${NO_COLOR}"
            ;;
        *)
            echo -e "${message}"
            ;;
    esac
}