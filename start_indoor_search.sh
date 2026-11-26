#!/usr/bin/env bash
# =============================================================================
# indoor_search 系统自动化启动脚本 (工程级重构版)
# 功能: tmux 多窗口可观测启动, 进程组管理, 完整 stop, 健康检查
#
# 用法:cd ~/PX4-Autopilot
#   ./scripts/start_indoor_search.sh full        # 完整启动 (含编译)
#   ./scripts/start_indoor_search.sh no-compile  # 快速启动 (跳过编译)
#   ./scripts/start_indoor_search.sh stop        # 完全停止所有组件
#   ./scripts/start_indoor_search.sh status      # 查看当前状态
#   ./scripts/start_indoor_search.sh health      # 检查 TF/话题/SLAM 健康状态
#   ./scripts/start_indoor_search.sh logs        # 查看最新日志路径
#   ./scripts/start_indoor_search.sh attach      # 进入 tmux session
#
# 依赖: tmux, micro-ros-agent, ros2(humble), slam_toolbox, pymavlink
# =============================================================================

set -o pipefail

# ──────────────────────────────────────────
# 颜色常量
# ──────────────────────────────────────────
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'
RED='\033[0;31m';   CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ──────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────
WORKSPACE="/home/ubuntu/PX4-Autopilot"
ROS2_WS="${WORKSPACE}/ros2_ws"
# SLAM_MODE: "full"（默认，启用 loop closing）| "debug"（禁用 loop closing，排查地图问题）
# 使用方法：SLAM_MODE=debug ./start_indoor_search.sh start
SLAM_MODE="${SLAM_MODE:-full}"
# DEBUG_SAVE_MAP: "0"（默认）| "1"（启用详细地图调试图保存，每5s输出一次）
# 使用方法：DEBUG_SAVE_MAP=1 ./start_indoor_search.sh full
DEBUG_SAVE_MAP="${DEBUG_SAVE_MAP:-0}"
_SLAM_PARAMS_FULL="${ROS2_WS}/install/share/indoor_search/config/slam_toolbox_params.yaml"
_SLAM_PARAMS_DEBUG="${ROS2_WS}/install/share/indoor_search/config/slam_toolbox_params_debug.yaml"
if [[ "${SLAM_MODE}" == "debug" ]]; then
    SLAM_PARAMS="${_SLAM_PARAMS_DEBUG}"
    echo "[SLAM_MODE] 调试模式：使用 slam_toolbox_params_debug.yaml (do_loop_closing=false)"
else
    SLAM_PARAMS="${_SLAM_PARAMS_FULL}"
fi

# DEBUG_SAVE_MAP 启动信息
if [[ "${DEBUG_SAVE_MAP}" == "1" ]]; then
    echo "[DEBUG_SAVE_MAP] ★ 详细地图调试模式已启用："
    echo "  → map_save_interval=5s（每5秒保存一次地图PNG/NPY）"
    echo "  → 日志目录: ${STATE_DIR}/logs/latest/"
    echo "  → 查看: watch -n 5 ls -la ${STATE_DIR}/logs/latest/map_output/"
fi

STATE_DIR="/home/ubuntu/PX4-Autopilot/start_log"
PID_DIR="${STATE_DIR}/pids"
LOG_BASE="${STATE_DIR}/logs"
LATEST_LINK="${LOG_BASE}/latest"

TMUX_SESSION="indoor_search"

# tmux window 名称
WIN_AGENT="1:agent"
WIN_PX4="2:px4"
WIN_BRIDGE="3:bridge"
WIN_SLAM="4:slam"
WIN_NODES="5:nodes"
WIN_MONITOR="6:monitor"

# ──────────────────────────────────────────
# ROS2 公共环境 (每个 tmux pane 里注入)
# ──────────────────────────────────────────
ROS_ENV="
export FASTRTPS_DEFAULT_PROFILES_FILE=/dev/null
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export GAZEBO_PLUGIN_PATH=${WORKSPACE}/build/px4_sitl_default/build_gazebo-classic:/opt/ros/humble/lib
export LD_LIBRARY_PATH=/opt/ros/humble/lib:\${LD_LIBRARY_PATH:-}
source /opt/ros/humble/setup.bash
source ~/px4_ros_com_ws/install/setup.bash
source ${ROS2_WS}/install/setup.bash
"

# ──────────────────────────────────────────
# 打印函数
# ──────────────────────────────────────────
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[✓]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[✗]${NC}    $*" >&2; }
section() { echo -e "\n${CYAN}${BOLD}══ $* ══${NC}"; }

# ──────────────────────────────────────────
# 初始化目录和日志
# ──────────────────────────────────────────
init_dirs() {
    TS=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${LOG_BASE}/${TS}"
    mkdir -p "${LOG_DIR}" "${PID_DIR}"
    # 更新 latest 软链接
    ln -sfn "${LOG_DIR}" "${LATEST_LINK}"
    info "日志目录: ${LOG_DIR}"
    info "快捷路径: ${LATEST_LINK}"
}

# ──────────────────────────────────────────
# 工具: 等待话题出现 (最多 timeout 秒)
# wait_for_topic <topic> <timeout_sec> <pane_hint>
# ──────────────────────────────────────────
wait_for_topic() {
    local topic="$1" timeout="$2" hint="${3:-}"
    local elapsed=0
    info "等待话题 ${topic} (最长 ${timeout}s)..."
    while ! ros2_eval "ros2 topic list 2>/dev/null | grep -qx '${topic}'"; do
        sleep 2; elapsed=$((elapsed + 2))
        if (( elapsed >= timeout )); then
            err "等待 ${topic} 超时! ${hint:+查看: $hint}"
            return 1
        fi
        (( elapsed % 10 == 0 )) && info "  仍在等待 ${topic}... ${elapsed}s/${timeout}s"
    done
    ok "话题 ${topic} 已就绪"
}

# wait_for_hz <topic> <min_hz> <timeout_sec> <pane_hint>
# 每轮：timeout 8s 采集 + sleep 2s 间隔 = ~10s/轮，elapsed 正确累加
wait_for_hz() {
    local topic="$1" min_hz="$2" timeout="$3" hint="${4:-}"
    local elapsed=0
    info "等待 ${topic} 频率 ≥ ${min_hz} Hz (最长 ${timeout}s)..."
    while true; do
        local hz_raw hz
        # 取最后一行 average rate（8s 内可能打印多行，tail -1 保留最终稳定值）
        hz_raw=$(ros2_eval "timeout 8 ros2 topic hz ${topic} 2>/dev/null \
            | grep 'average rate' | tail -1 | awk '{print \$3}' | tr -d ':'" \
            2>/dev/null || true)
        # 剥掉空白/换行，确保 awk 收到单一数字
        hz=$(echo "${hz_raw}" | tr -d '[:space:]')
        hz="${hz:-0}"
        if awk "BEGIN{exit !(${hz}+0 >= ${min_hz})}"; then
            ok "${topic} 频率 = ${hz} Hz ≥ ${min_hz} Hz"
            return 0
        fi
        sleep 2; elapsed=$((elapsed + 10))   # 8s(timeout) + 2s(sleep)
        if (( elapsed >= timeout )); then
            err "${topic} 频率未达 ${min_hz} Hz (当前 ${hz} Hz)! ${hint:+查看: $hint}"
            return 1
        fi
        (( elapsed % 20 == 0 )) && info "  当前 ${topic} = ${hz} Hz, 继续等待..."
    done
}

# wait_for_string_in_log <logfile> <pattern> <timeout_sec>
wait_for_string_in_log() {
    local logfile="$1" pattern="$2" timeout="$3"
    local elapsed=0
    info "等待日志 $(basename ${logfile}) 出现: '${pattern}' (最长 ${timeout}s)..."
    while ! grep -q "${pattern}" "${logfile}" 2>/dev/null; do
        sleep 2; elapsed=$((elapsed + 2))
        if (( elapsed >= timeout )); then
            err "超时! '${pattern}' 未在 ${logfile} 中找到"
            err "查看日志: tail -50 ${logfile}"
            return 1
        fi
        (( elapsed % 20 == 0 )) && info "  等待中... ${elapsed}s/${timeout}s"
    done
    ok "找到: '${pattern}'"
}

# ──────────────────────────────────────────
# 在已 source 好环境的子 shell 执行命令
# ──────────────────────────────────────────
ros2_eval() {
    bash -c "
        source /opt/ros/humble/setup.bash 2>/dev/null
        source ~/px4_ros_com_ws/install/setup.bash 2>/dev/null
        source ${ROS2_WS}/install/setup.bash 2>/dev/null
        export FASTRTPS_DEFAULT_PROFILES_FILE=/dev/null
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
        $*
    " 2>/dev/null
}

# ──────────────────────────────────────────
# 保存 PID/PGID
# ──────────────────────────────────────────
save_pid() {
    local name="$1" pid="$2"
    echo "${pid}" > "${PID_DIR}/${name}.pid"
    # 进程组 ID (子进程启动时已用 set -m / setsid，PGID = PID)
    local pgid
    pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ') || pgid="${pid}"
    echo "${pgid}" > "${PID_DIR}/${name}.pgid"
}

read_pidfile() {
    local file="$1"
    [[ -f "${file}" ]] && cat "${file}" || echo ""
}

# ──────────────────────────────────────────
# 优雅 kill 一个进程组
# kill_group <name> [timeout_sec]
# ──────────────────────────────────────────
kill_group() {
    local name="$1" timeout="${2:-8}"
    local pgid pid
    pgid=$(read_pidfile "${PID_DIR}/${name}.pgid")
    pid=$(read_pidfile  "${PID_DIR}/${name}.pid")

    if [[ -z "${pgid}" && -z "${pid}" ]]; then
        return 0
    fi

    local target="${pgid:-$pid}"
    info "停止 ${name} (PGID=${pgid:-?}, PID=${pid:-?})..."

    # 1) 先 SIGINT (让进程自己清理)
    [[ -n "${pgid}" ]] && kill -INT  -- "-${pgid}" 2>/dev/null || true
    [[ -n "${pid}"  ]] && kill -INT     "${pid}"   2>/dev/null || true
    sleep 2

    # 2) SIGTERM
    [[ -n "${pgid}" ]] && kill -TERM -- "-${pgid}" 2>/dev/null || true
    [[ -n "${pid}"  ]] && kill -TERM    "${pid}"   2>/dev/null || true

    # 3) 等待, 超时后 SIGKILL
    local waited=0
    while (( waited < timeout )); do
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi
        sleep 1; (( waited++ ))
    done
    [[ -n "${pgid}" ]] && kill -KILL -- "-${pgid}" 2>/dev/null || true
    [[ -n "${pid}"  ]] && kill -KILL    "${pid}"   2>/dev/null || true

    rm -f "${PID_DIR}/${name}.pid" "${PID_DIR}/${name}.pgid"
    ok "${name} 已停止"
}

# ──────────────────────────────────────────
# 在 tmux pane 中运行命令 (输出 tee 到日志)
# tmux_run <window_name> <logfile> <cmd>
# ──────────────────────────────────────────
tmux_run() {
    local win="$1" logfile="$2" cmd="$3"
    # 注入 ROS 环境 + stdbuf line-buffer + tee 日志
    local full_cmd="${ROS_ENV}
set -m
stdbuf -oL -eL bash -c '${cmd}' 2>&1 | tee -a '${logfile}'
"
    tmux send-keys -t "${TMUX_SESSION}:${win}" "${full_cmd}" Enter
}

# ──────────────────────────────────────────
# 获取 tmux pane 内最后启动进程的 PID
# 等待进程出现后记录
# get_tmux_pid <win> <match_pattern>
# ──────────────────────────────────────────
get_background_pid() {
    local pattern="$1" timeout="${2:-15}"
    local elapsed=0
    while (( elapsed < timeout )); do
        local pid
        pid=$(pgrep -f "${pattern}" | head -1)
        if [[ -n "${pid}" ]]; then echo "${pid}"; return 0; fi
        sleep 1; (( elapsed++ ))
    done
    echo ""
}

# ──────────────────────────────────────────
# 创建/刷新 tmux session
# ──────────────────────────────────────────
setup_tmux() {
    # 若已有同名 session, 先销毁
    tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
    sleep 1

    # 新建 session, 默认窗口命名为 agent
    # ★ 修复: tmux base-index 默认=0, new-session的第一个窗口是:0 不是:1
    #         -n "agent" 已完成命名，删除多余且错误的 rename-window :1 命令
    tmux new-session -d -s "${TMUX_SESSION}" -n "agent" -x 220 -y 50

    # 创建其余窗口
    tmux new-window -t "${TMUX_SESSION}" -n "px4"
    tmux new-window -t "${TMUX_SESSION}" -n "bridge"
    tmux new-window -t "${TMUX_SESSION}" -n "slam"
    tmux new-window -t "${TMUX_SESSION}" -n "nodes"
    tmux new-window -t "${TMUX_SESSION}" -n "monitor"

    ok "tmux session '${TMUX_SESSION}' 已创建 (6 windows)"
}

# ──────────────────────────────────────────
# 步骤 0: 编译
# ──────────────────────────────────────────
step_compile() {
    section "步骤 0: 编译 indoor_search"
    local log="${LOG_DIR}/compile.log"
    info "日志: ${log}"

    (
        cd "${ROS2_WS}"
        source /opt/ros/humble/setup.bash
        stdbuf -oL -eL colcon build \
            --symlink-install --merge-install \
            --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            2>&1 | tee "${log}"
    )
    local rc=$?
    if (( rc != 0 )); then
        err "编译失败! 查看: tail -50 ${log}"
        exit 1
    fi
    ok "编译完成"
}

# ──────────────────────────────────────────
# 步骤 1: micro-ROS Agent
# ──────────────────────────────────────────
step_agent() {
    section "步骤 1: micro-ROS Agent"
    local log="${LOG_DIR}/agent.log"

    tmux_run "agent" "${log}" \
        "exec micro-ros-agent udp4 --port 8888 -v6"

    # 等待进程出现并记录 PID
    local pid
    pid=$(get_background_pid "micro-ros-agent" 10)
    if [[ -z "${pid}" ]]; then
        err "micro-ROS Agent 未启动! 查看 tmux pane '${TMUX_SESSION}:agent' 或 ${log}"
        exit 1
    fi
    save_pid "agent" "${pid}"
    ok "micro-ROS Agent PID=${pid}, 日志: ${log}"
    sleep 2
}

# ──────────────────────────────────────────
# 步骤 2: PX4 SITL + Gazebo
# ──────────────────────────────────────────
step_px4() {
    section "步骤 2: PX4 SITL + Gazebo (maze)"
    local log="${LOG_DIR}/px4.log"

    # 清理旧临时文件
    rm -rf /tmp/px4* 2>/dev/null || true

    # PX4_SITL_WORLD=maze 命令不变
    # ★ 修复：加入 GAZEBO_MODEL_DATABASE_URI= 禁用 Gazebo Classic 在线模型数据库查询。
    #   根因：Gazebo Classic 加载任何 model:// URI 时，会先通过 HTTP 代理访问
    #         http://models.gazebosim.org/ 获取模型列表。代理 (127.0.0.1:7890) 不稳定时
    #         此请求挂起，gzserver 停在 "Loading world file" 不前进，PX4 进程永远不启动。
    #   设为空字符串后: Gazebo 只搜索本地 GAZEBO_MODEL_PATH，不发起任何网络连接。
    tmux_run "px4" "${log}" \
        "cd ${WORKSPACE} && PX4_SITL_WORLD=maze VERBOSE_SIM=1 GAZEBO_MODEL_DATABASE_URI= make px4_sitl gazebo"

    # 等待 PX4 出现 pxh> 提示符 (最长 240s，加载 maze.world + PX4 初始化需要时间)
    wait_for_string_in_log "${log}" "pxh>" 240 || {
        err "PX4 启动失败! 查看: ${WIN_PX4} pane 或 tail -50 ${log}"
        exit 1
    }

    # 额外等待 Gazebo 稳定
    sleep 5

    # 等待 uxrce_dds_client 起来 (让 /fmu 话题出现)
    info "等待 PX4→ROS2 DDS 桥接 (/fmu topics)..."
    local elapsed=0
    while ! ros2_eval "ros2 topic list 2>/dev/null | grep -q '/fmu/out/vehicle_local_position'"; do
        sleep 3; elapsed=$((elapsed+3))
        if (( elapsed > 90 )); then
            err "PX4 DDS 桥接超时! 检查 pxh> 是否有 uxrce_dds_client 错误"
            err "查看: ${WIN_PX4} pane 或 tail -50 ${log}"
            # 不 exit, 让用户继续观察
            break
        fi
    done
    ok "PX4 SITL + Gazebo 就绪"

    # 记录 PID (make 子进程 px4 的 PGID 即整个进程组)
    local px4_pid
    px4_pid=$(pgrep -x "px4" | head -1 || pgrep -f "px4_sitl" | head -1 || echo "")
    if [[ -n "${px4_pid}" ]]; then
        save_pid "px4" "${px4_pid}"
    fi
    local gz_pid; gz_pid=$(pgrep -x "gzserver" | head -1 || echo "")
    [[ -n "${gz_pid}" ]] && save_pid "gz" "${gz_pid}"

    # 自动设置 COM_OF_LOSS_T=1.5
    _set_px4_param "COM_OF_LOSS_T" "1.5"
}

# 用 pymavlink 自动设置 PX4 参数 (超时 20s 后放弃)
_set_px4_param() {
    local param="$1" val="$2"
    info "尝试自动设置 ${param}=${val} (via pymavlink)..."
    local pyfile; pyfile=$(mktemp /tmp/set_px4_param_XXXXXX.py)
    cat > "${pyfile}" <<PYEOF
import sys, time
try:
    from pymavlink import mavutil
    c = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    c.wait_heartbeat(timeout=12)
    c.mav.param_set_send(
        c.target_system, c.target_component,
        b'${param}', float(${val}),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
    if timeout 20 python3 "${pyfile}" 2>/dev/null; then
        ok "${param}=${val} 已自动设置"
    else
        warn "自动设置 ${param} 失败 (pymavlink 未连接?), 请手动执行:"
        warn "  mavproxy.py --master=udp:127.0.0.1:14550 --cmd='param set ${param} ${val};quit'"
    fi
    rm -f "${pyfile}"
}

# ──────────────────────────────────────────
# 步骤 3: px4_slam_bridge_node
# ──────────────────────────────────────────
step_bridge() {
    section "步骤 3: px4_slam_bridge_node"
    local log="${LOG_DIR}/bridge.log"

    # SLAM_SCAN_GATE 控制是否启用稳定门控（默认 1=启用）
    # 调试时可关闭：SLAM_SCAN_GATE=0 ./start_indoor_search.sh full
    SLAM_SCAN_GATE="${SLAM_SCAN_GATE:-1}"
    local _gate_enabled="true"
    [[ "${SLAM_SCAN_GATE}" == "0" ]] && _gate_enabled="false"
    info "SLAM_SCAN_GATE=${SLAM_SCAN_GATE} → slam_scan_gate_enabled=${_gate_enabled}"

    tmux_run "bridge" "${log}" \
        "exec ros2 run indoor_search px4_slam_bridge_node.py --ros-args \
          -p use_sim_time:=true \
          -p fmu_prefix:=/fmu \
          -p lidar_topic_in:=/px4/lidar_controller/out \
          -p lidar_mount_yaw_deg:=0.0 \
          -p scan_raw_topic:=/scan_raw \
          -p scan_slam_topic:=/scan \
          -p slam_scan_gate_enabled:=${_gate_enabled} \
          -p gate_roll_deg:=5.0 \
          -p gate_pitch_deg:=5.0 \
          -p gate_xy_speed_mps:=0.15 \
          -p gate_z_speed_mps:=0.08 \
          -p gate_yaw_rate_dps:=15.0 \
          -p gate_hold_sec:=0.3"

    local pid
    pid=$(get_background_pid "px4_slam_bridge_node" 15)
    if [[ -z "${pid}" ]]; then
        err "bridge 启动失败! 查看: ${WIN_BRIDGE} pane 或 ${log}"
        exit 1
    fi
    save_pid "bridge" "${pid}"
    ok "bridge PID=${pid}, 日志: ${log}"

    # /scan_raw 始终发布（gate 关闭时 /scan 可能 Hz=0，但 /scan_raw 保持正常频率）
    wait_for_topic "/scan_raw" 60 "pane ${WIN_BRIDGE} 或 ${log}" || true
    wait_for_hz "/scan_raw" 5 45 "pane ${WIN_BRIDGE} 或 ${log}" || {
        warn "/scan_raw 频率不足, 检查 lidar 话题: ros2 topic hz /px4/lidar_controller/out"
    }
}

# ──────────────────────────────────────────
# 步骤 4: slam_toolbox
# ──────────────────────────────────────────
step_slam() {
    section "步骤 4: slam_toolbox"
    local log="${LOG_DIR}/slam.log"

    # 确认 params 文件存在
    if [[ ! -f "${SLAM_PARAMS}" ]]; then
        err "slam_toolbox params 文件不存在: ${SLAM_PARAMS}"
        err "请先执行编译 (colcon build)"
        exit 1
    fi

    tmux_run "slam" "${log}" \
        "exec ros2 run slam_toolbox async_slam_toolbox_node --ros-args \
          --params-file '${SLAM_PARAMS}' \
          -p use_sim_time:=true"

    local pid
    pid=$(get_background_pid "async_slam_toolbox_node" 20)
    if [[ -z "${pid}" ]]; then
        err "slam_toolbox 启动失败! 查看: ${WIN_SLAM} pane 或 ${log}"
        exit 1
    fi
    save_pid "slam" "${pid}"
    ok "slam_toolbox PID=${pid}, 日志: ${log}"

    # 等待 /map 出现
    wait_for_topic "/map" 90 "pane ${WIN_SLAM} 或 ${log}" || {
        warn "/map 话题未出现, 可能 slam_toolbox 还未建图 (正常, 继续)"
    }
}

# ──────────────────────────────────────────
# 步骤 5: indoor_search 业务节点
# ──────────────────────────────────────────
step_nodes() {
    section "步骤 5: indoor_search 业务节点"
    local log="${LOG_DIR}/nodes.log"

    # 所有节点在 nodes window 里顺序启动 (用 & 后台, 最后 wait)
    # stdbuf + tee 都接到同一个 log, 方便观察
    tmux_run "nodes" "${log}" "
# ── target_search ────────────────────────────────────────────────
# DEBUG_SAVE_MAP=${DEBUG_SAVE_MAP}: 1=每5s保存详细调试图，0=默认30s
_MAP_SAVE_INTERVAL=\$([ '${DEBUG_SAVE_MAP}' = '1' ] && echo 5.0 || echo 30.0)
echo \"[NODES] DEBUG_SAVE_MAP=${DEBUG_SAVE_MAP} → map_save_interval=\${_MAP_SAVE_INTERVAL}s\"
ros2 run indoor_search target_search_node.py --ros-args \\
  -p use_sim_time:=true \\
  -p fmu_prefix:=/fmu \\
  -p slam_map_topic:=/map \\
  -p map_bound_xmin:=-8.0 \\
  -p map_bound_xmax:=8.0 \\
  -p map_bound_ymin:=-8.0 \\
  -p map_bound_ymax:=8.0 \\
  -p map_res:=0.1 \\
  -p display_fixed_bounds:=true \\
  -p display_xmin:=-8.0 \\
  -p display_xmax:=8.0 \\
  -p display_ymin:=-8.0 \\
  -p display_ymax:=8.0 \\
  -p vis_enabled:=true \\
  -p vis_mode:=global \\
  -p map_save_interval:=\${_MAP_SAVE_INTERVAL} &
TS_PID=\$!
echo \"[NODES] target_search PID=\${TS_PID}\"
sleep 2

# ── local_mpc ────────────────────────────────────────────────────
ros2 run indoor_search local_mpc_node.py --ros-args \
  -p use_sim_time:=true \
  -p fmu_prefix:=/fmu &
MPC_PID=\$!
echo \"[NODES] local_mpc PID=\${MPC_PID}\"

# ── apf_node ─────────────────────────────────────────────────────
ros2 run indoor_search apf_node.py --ros-args \
  -p use_sim_time:=true \
  -p fmu_prefix:=/fmu &
APF_PID=\$!
echo \"[NODES] apf_node PID=\${APF_PID}\"

# ── global_planner ───────────────────────────────────────────────
ros2 run indoor_search global_planner_node.py --ros-args \
  -p use_sim_time:=true &
GP_PID=\$!
echo \"[NODES] global_planner PID=\${GP_PID}\"

echo \"[NODES] 所有业务节点已后台启动\"
wait
"

    # 等待关键节点进程出现并记录
    sleep 4
    local ts_pid
    ts_pid=$(get_background_pid "target_search_node" 20)
    if [[ -z "${ts_pid}" ]]; then
        warn "target_search_node 未检测到, 查看: ${WIN_NODES} pane 或 ${log}"
    else
        save_pid "search" "${ts_pid}"
        ok "target_search PID=${ts_pid}"
    fi

    # ── 等待 command_loop 第一帧启动（日志检测，不依赖 DDS 发现延迟）──
    wait_for_string_in_log "${log}" "\[CMD-LOOP-READY\]" 60 || {
        err "command_loop 未在 60s 内启动! 查看: ${log}"
        err "提示: 检查 use_sim_time / /clock 话题 / ROS 环境"
    }

    # ── 等待位姿初始化完成（确认传感器链路正常）──
    wait_for_string_in_log "${log}" "\[INIT\] pose initialized" 120 || {
        warn "位姿初始化超时 (120s)，传感器链路可能异常，仍继续尝试解锁"
        warn "查看日志: tail -50 ${log}"
    }
}

# ──────────────────────────────────────────
# 步骤 6: monitor 窗口 (方便观察)
# ──────────────────────────────────────────
step_monitor() {
    section "步骤 6: monitor 窗口"
    tmux send-keys -t "${TMUX_SESSION}:monitor" \
        "${ROS_ENV}
echo '=== 节点列表 ==='; watch -n 2 ros2 node list" Enter
}

# ──────────────────────────────────────────
# 步骤 7: 自动解锁 + 切换 Offboard
# ──────────────────────────────────────────
step_arm_offboard() {
    section "步骤 7: 自动解锁 + Offboard"
    local log="${LOG_DIR}/arm_offboard.log"
    local script="${WORKSPACE}/Tools/sitl_arm_offboard.py"

    if [[ ! -f "${script}" ]]; then
        err "找不到 ${script}"
        warn "请手动解锁: 在 pxh> 输入 commander arm; commander mode offboard"
        return 1
    fi

    info "调用 sitl_arm_offboard.py --no-confirm ..."
    info "日志: ${log}"

    # 调用脚本; 输出同时打印到终端和日志文件
    if python3 "${script}" --no-confirm --log-dir "${LOG_DIR}" 2>&1 | tee "${log}"; then
        ok "自动解锁 + Offboard 切换成功"
    else
        warn "自动解锁失败 (exit=$?), 查看日志: tail -50 ${log}"
        warn "可手动在 pxh> 终端执行: commander arm; commander mode offboard"
    fi
}

# ──────────────────────────────────────────
# 最终提示
# ──────────────────────────────────────────
print_final() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                系统已完全启动并自动解锁！                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${CYAN}进入 tmux 查看各组件实时输出:${NC}"
    echo -e "  ${BOLD}tmux attach -t ${TMUX_SESSION}${NC}"
    echo ""
    echo -e "${CYAN}tmux 窗口导航:${NC}"
    echo -e "  Ctrl-b + 1  →  micro-ROS Agent"
    echo -e "  Ctrl-b + 2  →  PX4 SITL / Gazebo"
    echo -e "  Ctrl-b + 3  →  px4_slam_bridge"
    echo -e "  Ctrl-b + 4  →  slam_toolbox"
    echo -e "  Ctrl-b + 5  →  indoor_search 节点"
    echo -e "  Ctrl-b + 6  →  monitor (ros2 node list)"
    echo ""
    echo -e "${CYAN}日志目录 (${LOG_DIR}):${NC}"
    echo -e "  agent.log / px4.log / bridge.log / slam.log / nodes.log"
    echo -e "  ${YELLOW}arm_offboard.log${NC}  ← 解锁/Offboard 切换日志"
    echo -e "  快捷路径: ${LATEST_LINK}/"
    echo ""
    echo -e "${YELLOW}如解锁失败, 可手动在 pxh> 终端执行:${NC}"
    echo -e "    ${CYAN}commander arm${NC}"
    echo -e "    ${CYAN}commander mode offboard${NC}"
    echo ""
    echo -e "${YELLOW}停止: ${BOLD}./scripts/start_indoor_search.sh stop${NC}"
}

# ──────────────────────────────────────────
# stop: 完全终止所有组件
# ──────────────────────────────────────────
cmd_stop() {
    section "停止所有组件"

    # 1) kill tmux session (会关闭所有 pane 内的前台进程)
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        info "关闭 tmux session '${TMUX_SESSION}'..."
        tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
        sleep 2
    fi

    # 2) 按反向顺序, 用 pidfile 进程组 kill
    for comp in search slam bridge px4 gz agent; do
        kill_group "${comp}" 8
    done

    # 3) 兜底 pkill (确保无残留)
    info "兜底清理残留进程..."
    pkill -KILL -f "async_slam_toolbox_node" 2>/dev/null || true
    pkill -KILL -f "px4_slam_bridge_node"    2>/dev/null || true
    pkill -KILL -f "target_search_node"      2>/dev/null || true
    pkill -KILL -f "local_mpc_node"          2>/dev/null || true
    pkill -KILL -f "apf_node"               2>/dev/null || true
    pkill -KILL -f "global_planner_node"     2>/dev/null || true
    pkill -KILL -f "offboard_commander"      2>/dev/null || true
    pkill -KILL -f "sitl_arm_offboard"       2>/dev/null || true
    pkill -KILL -f "micro-ros-agent"         2>/dev/null || true
    pkill -KILL -x "px4"                     2>/dev/null || true
    pkill -KILL -x "gzserver"                2>/dev/null || true
    pkill -KILL -x "gzclient"                2>/dev/null || true
    # Gazebo 新版进程
    pkill -KILL -x "gz"                      2>/dev/null || true
    pkill -KILL -f "gz sim"                  2>/dev/null || true
    # make 衍生的中间进程
    pkill -KILL -f "px4_sitl"               2>/dev/null || true

    # 4) 清理 PX4 临时文件
    rm -rf /tmp/px4* 2>/dev/null || true

    # 5) 验证
    sleep 2
    local residual=0
    for pattern in "micro-ros-agent" "gzserver" "gzclient" "px4_slam_bridge_node" \
                   "async_slam_toolbox_node" "target_search_node"; do
        if pgrep -f "${pattern}" >/dev/null 2>&1; then
            warn "仍有残留进程: ${pattern} ($(pgrep -f ${pattern}))"
            residual=1
        fi
    done

    if (( residual == 0 )); then
        ok "所有组件已完全停止"
    else
        warn "部分进程可能残留, 执行以下命令彻底清理:"
        echo "  sudo kill -9 \$(pgrep -f 'gzserver|px4|micro-ros|slam_toolbox|indoor_search')"
    fi

    # 清理 pidfiles
    rm -f "${PID_DIR}"/*.pid "${PID_DIR}"/*.pgid 2>/dev/null || true
}

# ──────────────────────────────────────────
# status: 显示当前状态
# ──────────────────────────────────────────
cmd_status() {
    section "系统状态"

    echo -e "\n${BLUE}[进程]${NC}"
    local procs=("micro-ros-agent:agent" "gzserver:Gazebo" "px4_slam_bridge_node:bridge"
                 "async_slam_toolbox_node:slam" "target_search_node:search")
    for entry in "${procs[@]}"; do
        local pat="${entry%%:*}" label="${entry##*:}"
        if pgrep -f "${pat}" >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} ${label} (PID: $(pgrep -f ${pat} | head -1))"
        else
            echo -e "  ${RED}✗${NC} ${label}"
        fi
    done

    echo -e "\n${BLUE}[ROS2 节点]${NC}"
    ros2_eval "ros2 node list 2>/dev/null" | sed 's/^/  /' || echo "  (ros2 不可用)"

    echo -e "\n${BLUE}[关键话题 Publisher count]${NC}"
    for t in /scan /map /fmu/in/offboard_control_mode /fmu/in/trajectory_setpoint; do
        local cnt
        cnt=$(ros2_eval "ros2 topic info ${t} 2>/dev/null | grep 'Publisher count' | awk '{print \$NF}'" || echo "?")
        echo -e "  ${t}: ${cnt} publisher(s)"
    done

    echo -e "\n${BLUE}[tmux session]${NC}"
    tmux has-session -t "${TMUX_SESSION}" 2>/dev/null \
        && echo -e "  ${GREEN}✓${NC} session '${TMUX_SESSION}' 运行中  →  tmux attach -t ${TMUX_SESSION}" \
        || echo -e "  ${RED}✗${NC} session '${TMUX_SESSION}' 未运行"

    echo -e "\n${BLUE}[日志]${NC}"
    if [[ -d "${LATEST_LINK}" ]]; then
        echo "  最新日志: $(realpath ${LATEST_LINK})"
        ls -lh "${LATEST_LINK}"/*.log 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}' || true
    else
        echo "  无日志"
    fi
}

# ──────────────────────────────────────────
# logs: 显示日志路径
# ──────────────────────────────────────────
cmd_logs() {
    if [[ ! -d "${LATEST_LINK}" ]]; then
        warn "还没有运行日志"; return
    fi
    local logdir; logdir=$(realpath "${LATEST_LINK}")
    echo -e "${CYAN}最新日志目录:${NC} ${logdir}"
    echo ""
    echo -e "${CYAN}实时跟踪命令:${NC}"
    for f in agent px4 bridge slam nodes compile; do
        local p="${logdir}/${f}.log"
        [[ -f "${p}" ]] && echo "  tail -f ${p}  # ${f}"
    done
    echo ""
    echo -e "${CYAN}合并查看最后 50 行:${NC}"
    echo "  tail -50 ${logdir}/*.log"
}

# ──────────────────────────────────────────
# attach: 进入 tmux
# ──────────────────────────────────────────
cmd_attach() {
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        exec tmux attach -t "${TMUX_SESSION}"
    else
        err "tmux session '${TMUX_SESSION}' 不存在, 请先启动系统"
        exit 1
    fi
}

# ──────────────────────────────────────────
# health: SLAM + TF + 话题健康检查
# ──────────────────────────────────────────
cmd_health() {
    section "系统健康检查 (约 30s)"

    local PASS=0 FAIL=0

    _hcheck_ok()  { echo -e "  ${GREEN}[PASS]${NC} $*"; (( PASS++ )); }
    _hcheck_warn(){ echo -e "  ${YELLOW}[WARN]${NC} $*"; }
    _hcheck_fail(){ echo -e "  ${RED}[FAIL]${NC} $*"; (( FAIL++ )); }

    # ── 环境检查 ──────────────────────────────────────────────────────
    echo -e "\n${BLUE}[0] ROS2 环境${NC}"
    if ! source /opt/ros/humble/setup.bash 2>/dev/null \
       || ! source "${ROS2_WS}/install/setup.bash" 2>/dev/null; then
        _hcheck_fail "ROS2 环境无法加载, 以下检查可能不准确"
    else
        _hcheck_ok  "ROS2 Humble 环境加载正常"
    fi

    # ── 进程存活 ──────────────────────────────────────────────────────
    echo -e "\n${BLUE}[1] 关键进程存活${NC}"
    local procs=(
        "micro-ros-agent:micro-ROS Agent"
        "gzserver:Gazebo Server"
        "px4_slam_bridge_node:PX4-SLAM Bridge"
        "async_slam_toolbox_node:slam_toolbox"
        "target_search_node:Target Search"
    )
    for entry in "${procs[@]}"; do
        local pat="${entry%%:*}" label="${entry##*:}"
        if pgrep -f "${pat}" >/dev/null 2>&1; then
            _hcheck_ok  "${label}  (PID $(pgrep -f ${pat} | head -1))"
        else
            _hcheck_fail "${label} 进程未运行"
        fi
    done

    # ── TF 链完整性 ───────────────────────────────────────────────────
    echo -e "\n${BLUE}[2] TF 链完整性 (超时 5s/链)${NC}"
    local tf_pairs=(
        "odom:base_link:EKF odom→base_link (Bridge)"
        "base_link:laser_link:激光安装 base_link→laser_link (Static)"
        "map:odom:SLAM 闭环 map→odom (slam_toolbox)"
    )
    for entry in "${tf_pairs[@]}"; do
        IFS=':' read -r from to label <<< "${entry}"
        local result
        result=$(timeout 5 bash -c "
            source /opt/ros/humble/setup.bash 2>/dev/null
            source ${ROS2_WS}/install/setup.bash 2>/dev/null
            ros2 run tf2_ros tf2_echo ${from} ${to} 2>&1 | head -6
        " 2>&1 || true)
        if echo "${result}" | grep -q 'Translation\|transform\|- Translation'; then
            # 提取最新平移
            local trans
            trans=$(echo "${result}" | grep -A1 'Translation' | tail -1 | sed 's/^ *//')
            _hcheck_ok  "${label}  ${trans}"
        else
            _hcheck_fail "${label}  → TF 不可用 (slam_toolbox 未建图 或 bridge 未发布)"
        fi
    done

    # ── 话题频率 ──────────────────────────────────────────────────────
    echo -e "\n${BLUE}[3] 关键话题频率 (各采样 8s)${NC}"
    # 格式: topic:预期最低Hz:说明
    local topic_checks=(
        "/scan:8.0:激光雷达扫描 (RPLidar A3 ~15Hz)"
        "/map:0.4:/map OccupancyGrid (slam_toolbox 1Hz)"
        "/fmu/out/vehicle_local_position:10.0:PX4 位置 EKF 输出"
        "/fmu/in/trajectory_setpoint:0.1:规划器速度指令"
    )
    for entry in "${topic_checks[@]}"; do
        IFS=':' read -r topic min_hz label <<< "${entry}"
        local hz_raw
        hz_raw=$(timeout 10 bash -c "
            source /opt/ros/humble/setup.bash 2>/dev/null
            source ${ROS2_WS}/install/setup.bash 2>/dev/null
            ros2 topic hz --window 20 ${topic} 2>/dev/null
        " 2>&1 | grep 'average rate' | tail -1 | awk '{print $3}' | tr -d ':')
        if [[ -z "${hz_raw}" ]]; then
            _hcheck_fail "${label}  → /下线/ 0 Hz (话题无发布者或无消息)"
        else
            local hz; hz=${hz_raw%.*}   # 取整
            local thresh; thresh=${min_hz%.*}
            if (( 10#${hz:-0} >= 10#${thresh:-0} )); then
                _hcheck_ok  "${label}  ${hz_raw} Hz"
            else
                _hcheck_warn "${label}  ${hz_raw} Hz  (低于期望 ${min_hz} Hz)"
            fi
        fi
    done

    # ── /map 时间戳新鲜度 ─────────────────────────────────────────────
    echo -e "\n${BLUE}[4] /map 元数据新鲜度${NC}"
    local map_info
    map_info=$(timeout 8 bash -c "
        source /opt/ros/humble/setup.bash 2>/dev/null
        source ${ROS2_WS}/install/setup.bash 2>/dev/null
        ros2 topic echo --once --field header /map 2>/dev/null
    " 2>&1 || true)
    if echo "${map_info}" | grep -q 'stamp\|sec:'; then
        local stamp; stamp=$(echo "${map_info}" | grep 'sec:' | head -1 | awk '{print $2}')
        _hcheck_ok  "/map 有效  stamp.sec=${stamp}"
        local w h res
        w=$(timeout 4 bash -c "
            source /opt/ros/humble/setup.bash 2>/dev/null
            source ${ROS2_WS}/install/setup.bash 2>/dev/null
            ros2 topic echo --once --field info.width  /map 2>/dev/null" 2>&1 || echo "?")
        h=$(timeout 4 bash -c "
            source /opt/ros/humble/setup.bash 2>/dev/null
            source ${ROS2_WS}/install/setup.bash 2>/dev/null
            ros2 topic echo --once --field info.height /map 2>/dev/null" 2>&1 || echo "?")
        res=$(timeout 4 bash -c "
            source /opt/ros/humble/setup.bash 2>/dev/null
            source ${ROS2_WS}/install/setup.bash 2>/dev/null
            ros2 topic echo --once --field info.resolution /map 2>/dev/null" 2>&1 || echo "?")
        echo -e "      地图尺寸: ${w}×${h}  分辨率: ${res} m/cell"
    else
        _hcheck_fail "/map 无响应 (slam_toolbox 未建图 或 /scan 未到达)"
    fi

    # ── Bridge TF 诊断日志 (从 stderr 采样) ──────────────────────────
    echo -e "\n${BLUE}[5] Bridge TF 诊断日志 (最近 5 条)${NC}"
    local bridge_log="${LATEST_LINK}/bridge.log"
    if [[ -f "${bridge_log}" ]]; then
        grep '\[BRIDGE-TF\]' "${bridge_log}" 2>/dev/null | tail -5 \
            | sed 's/^/  /' \
            || echo "  (暂无 [BRIDGE-TF] 诊断日志)"
        # 检查 yaw 偏差警告
        local yaw_warn_cnt
        yaw_warn_cnt=$(grep -c 'yaw_diff>5' "${bridge_log}" 2>/dev/null || echo 0)
        if (( yaw_warn_cnt > 0 )); then
            _hcheck_warn "发现 ${yaw_warn_cnt} 次 yaw_diff>5° 告警 → att_quat 与 heading 不一致"
        else
            _hcheck_ok  "无 yaw_diff>5° 告警 (heading 与 quaternion 一致)"
        fi
        # 检查 SCAN-GATE 状态
        local gate_line
        gate_line=$(grep '\[SCAN-GATE\]' "${bridge_log}" 2>/dev/null | tail -1)
        if [[ -n "${gate_line}" ]]; then
            echo "  最近 SCAN-GATE: ${gate_line}"
            if echo "${gate_line}" | grep -q 'open=True'; then
                _hcheck_ok  "SCAN-GATE 最近为 open（稳定门已开，SLAM 接收扫描正常）"
            else
                _hcheck_warn "SCAN-GATE 最近为 closed（飞行不稳定，SLAM 输入被门控）"
            fi
        else
            echo "  (暂无 [SCAN-GATE] 日志，节点刚启动或未到100帧)"
        fi
        # jump 告警
        local jump_cnt
        jump_cnt=$(grep -c '\[TF-JUMP\]' "${bridge_log}" 2>/dev/null || echo 0)
        (( jump_cnt > 0 )) && _hcheck_warn "发现 ${jump_cnt} 次 [TF-JUMP] → 检查 EKF/网络稳定性" \
                           || _hcheck_ok  "无 [TF-JUMP] 大跳变"
    else
        _hcheck_warn "bridge 日志不存在 (${bridge_log})"
    fi

    # ── slam_toolbox 日志关键字 ───────────────────────────────────────
    echo -e "\n${BLUE}[6] slam_toolbox 日志关键字${NC}"
    local slam_log="${LATEST_LINK}/slam.log"
    if [[ -f "${slam_log}" ]]; then
        # 检查已知错误
        local err_scan err_tf err_loop
        err_scan=$(grep -c -i 'scan_queue.*full\|queue is full' "${slam_log}" 2>/dev/null || echo 0)
        err_tf=$(grep -c 'TF Exception\|Failed to compute\|No transform' "${slam_log}" 2>/dev/null || echo 0)
        err_loop=$(grep -c -i 'loop.*closure.*rejected\|failed loop' "${slam_log}" 2>/dev/null || echo 0)
        local ok_init
        ok_init=$(grep -c 'Serialization library\|Initialized slam' "${slam_log}" 2>/dev/null || echo 0)
        [[ ${ok_init}   -gt 0 ]] && _hcheck_ok "slam_toolbox 初始化日志存在"
        [[ ${err_scan}  -gt 0 ]] && _hcheck_warn "scan_queue full ×${err_scan} 次 (考虑降低 scan_queue_size 或激光频率)"
        [[ ${err_tf}    -gt 0 ]] && _hcheck_warn "TF 异常 ×${err_tf} 次 (Bridge 可能滞后)"
        [[ ${err_loop}  -gt 0 ]] && _hcheck_warn "回环闭合拒绝 ×${err_loop} 次 (正常现象，数量大时调高阈值)"
        [[ ${ok_init} -eq 0 && ${err_scan} -eq 0 && ${err_tf} -eq 0 ]] \
            && _hcheck_warn "slam 日志未见初始化信息 (进程未启动或日志路径不符)"
    else
        _hcheck_warn "slam 日志不存在 (${slam_log})"
    fi

    # ── 汇总 ──────────────────────────────────────────────────────────
    echo ""
    echo -e "────────────────────────────────────────"
    if (( FAIL == 0 )); then
        echo -e "${GREEN}${BOLD}健康检查通过: ${PASS} 项 PASS, 0 项 FAIL${NC}"
    else
        echo -e "${RED}${BOLD}健康检查结果: ${PASS} 项 PASS, ${FAIL} 项 FAIL${NC}"
        echo -e "${YELLOW}修复建议:${NC}"
        echo "  FAIL-进程未运行  → ./scripts/start_indoor_search.sh full"
        echo "  FAIL-TF 不可用   → tmux attach -t indoor_search → Ctrl-b+3 查看 bridge 窗口"
        echo "  FAIL-/map 无响应 → tmux attach -t indoor_search → Ctrl-b+4 查看 slam 窗口"
    fi
}

# ──────────────────────────────────────────
# 主启动流程
# ──────────────────────────────────────────
do_start() {
    local compile="${1:-true}"

    init_dirs
    setup_tmux

    [[ "${compile}" == "true" ]] && step_compile

    step_agent
    step_px4
    step_bridge
    step_slam
    step_nodes
    step_monitor
    step_arm_offboard
    print_final
}

# ──────────────────────────────────────────
# 入口
# ──────────────────────────────────────────
case "${1:-help}" in
    full)
        do_start "true"
        ;;
    no-compile)
        do_start "false"
        ;;
    debug)
        # 调试模式：SLAM_MODE=debug（禁用 loop closing），自动完整启动+编译
        SLAM_MODE="debug"
        _SLAM_PARAMS_DEBUG="${ROS2_WS}/install/share/indoor_search/config/slam_toolbox_params_debug.yaml"
        SLAM_PARAMS="${_SLAM_PARAMS_DEBUG}"
        echo "[SLAM_MODE] debug 快捷命令：do_loop_closing=false，隔离地图突变/重影问题"
        do_start "true"
        ;;
    debug-nc)
        # 调试模式快速启动（跳过编译）
        SLAM_MODE="debug"
        _SLAM_PARAMS_DEBUG="${ROS2_WS}/install/share/indoor_search/config/slam_toolbox_params_debug.yaml"
        SLAM_PARAMS="${_SLAM_PARAMS_DEBUG}"
        echo "[SLAM_MODE] debug-nc 快捷命令：do_loop_closing=false，快速启动"
        do_start "false"
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    health)
        cmd_health
        ;;
    logs)
        cmd_logs
        ;;
    attach)
        cmd_attach
        ;;
    help|--help|-h)
        cat <<'EOF'
用法: ./scripts/start_indoor_search.sh <命令>

命令:
  full        完整启动 (colcon build + 所有组件)                [生产模式: loop closing 开启]
  no-compile  快速启动 (跳过编译, 直接启动组件)                  [生产模式]
  debug       调试启动 (colcon build + SLAM_MODE=debug)        [调试模式: loop closing 关闭]
  debug-nc    调试快速启动 (跳过编译 + SLAM_MODE=debug)         [调试模式]
  stop        完全停止所有组件 (等价于手动关闭所有终端)
  status      显示当前进程/话题/tmux 状态
  health      深度健康检查: 进程/TF链/话题Hz/地图/Bridge诊断日志 (~30s)
  logs        显示最新日志路径和 tail 命令
  attach      进入 tmux session (Ctrl-b + 数字 切换窗口)

SLAM_MODE 环境变量（高级）:
  full（默认） → slam_toolbox_params.yaml：启用 loop closing
  debug        → slam_toolbox_params_debug.yaml：禁用 loop closing
  示例: SLAM_MODE=debug ./scripts/start_indoor_search.sh no-compile

tmux 窗口布局:
  1:agent   micro-ROS Agent
  2:px4     PX4 SITL + Gazebo
  3:bridge  px4_slam_bridge_node
  4:slam    slam_toolbox
  5:nodes   所有 indoor_search 业务节点
  6:monitor ros2 node list 监控

确认 stop 成功:
  pgrep gzserver px4 micro-ros-agent    # 应无输出
  ros2 node list                         # 应无业务节点
  lsof -i :8888                          # 应无输出

快速排障:
  失败在哪步 → tmux attach -t indoor_search → Ctrl-b+数字 进对应 pane
  或: tail -50 /home/ubuntu/PX4-Autopilot/start_log/logs/latest/<component>.log
  地图突变/重影：先用 debug/debug-nc 命令隔离（禁用 loop closing），若正常则闭环参数需调整
EOF
        ;;
    *)
        err "未知命令: $1"
        echo "执行 $0 help 查看用法"
        exit 1
        ;;
esac
