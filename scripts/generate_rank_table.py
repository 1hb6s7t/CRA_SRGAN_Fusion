# Copyright 2024
# 昇腾HCCL配置文件生成工具
"""
生成昇腾分布式训练所需的rank_table配置文件

用法:
    python generate_rank_table.py --device_num 8 --output rank_table_8pcs.json
"""

import os
import json
import argparse
import subprocess


def get_device_info():
    """获取NPU设备信息"""
    try:
        result = subprocess.run(['npu-smi', 'info', '-l'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        print("警告: npu-smi命令未找到,将使用默认配置")
        return None


def generate_rank_table(device_num, server_id='localhost', start_ip='192.168.100.101'):
    """
    生成rank_table配置
    
    Args:
        device_num: 设备数量
        server_id: 服务器ID
        start_ip: 起始IP地址
    """
    # 解析起始IP
    ip_parts = start_ip.rsplit('.', 1)
    ip_prefix = ip_parts[0]
    ip_start = int(ip_parts[1])
    
    devices = []
    for i in range(device_num):
        device = {
            "device_id": str(i),
            "device_ip": f"{ip_prefix}.{ip_start + i}",
            "rank_id": str(i)
        }
        devices.append(device)
    
    rank_table = {
        "version": "1.0",
        "server_count": "1",
        "server_list": [
            {
                "server_id": server_id,
                "device": devices
            }
        ],
        "status": "completed"
    }
    
    return rank_table


def generate_multi_server_rank_table(server_configs):
    """
    生成多服务器rank_table配置
    
    Args:
        server_configs: [
            {"server_id": "server1", "devices": [0,1,2,3], "start_ip": "192.168.1.1"},
            {"server_id": "server2", "devices": [0,1,2,3], "start_ip": "192.168.2.1"},
        ]
    """
    server_list = []
    rank_id = 0
    
    for server in server_configs:
        ip_parts = server['start_ip'].rsplit('.', 1)
        ip_prefix = ip_parts[0]
        ip_start = int(ip_parts[1])
        
        devices = []
        for i, dev_id in enumerate(server['devices']):
            device = {
                "device_id": str(dev_id),
                "device_ip": f"{ip_prefix}.{ip_start + i}",
                "rank_id": str(rank_id)
            }
            devices.append(device)
            rank_id += 1
        
        server_entry = {
            "server_id": server['server_id'],
            "device": devices
        }
        server_list.append(server_entry)
    
    rank_table = {
        "version": "1.0",
        "server_count": str(len(server_configs)),
        "server_list": server_list,
        "status": "completed"
    }
    
    return rank_table


def main():
    parser = argparse.ArgumentParser(description='生成昇腾HCCL配置文件')
    
    parser.add_argument('--device_num', type=int, default=8,
                       help='设备数量 (1, 2, 4, 8)')
    parser.add_argument('--output', type=str, default='rank_table.json',
                       help='输出文件路径')
    parser.add_argument('--server_id', type=str, default='localhost',
                       help='服务器ID')
    parser.add_argument('--start_ip', type=str, default='192.168.100.101',
                       help='起始IP地址')
    parser.add_argument('--multi_server', action='store_true',
                       help='多服务器模式')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("HCCL配置文件生成")
    print("=" * 50)
    
    # 获取设备信息
    device_info = get_device_info()
    if device_info:
        print("检测到的NPU设备:")
        print(device_info)
    
    # 生成配置
    if args.multi_server:
        # 示例: 2台服务器,每台8卡
        print("多服务器模式")
        server_configs = [
            {"server_id": "server1", "devices": list(range(8)), "start_ip": "192.168.1.101"},
            {"server_id": "server2", "devices": list(range(8)), "start_ip": "192.168.2.101"},
        ]
        rank_table = generate_multi_server_rank_table(server_configs)
    else:
        print(f"单服务器模式, {args.device_num}卡")
        rank_table = generate_rank_table(
            args.device_num,
            args.server_id,
            args.start_ip
        )
    
    # 保存
    with open(args.output, 'w') as f:
        json.dump(rank_table, f, indent=4)
    
    print(f"\n配置文件已保存到: {args.output}")
    print("\n生成的配置:")
    print(json.dumps(rank_table, indent=2))
    
    print("\n使用方法:")
    print(f"  export RANK_TABLE_FILE={args.output}")
    print(f"  ./scripts/run_distribute_train_ascend.sh {args.output} {args.device_num} ...")


if __name__ == '__main__':
    main()

