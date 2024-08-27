import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from cleaner import categorize_ip_by_subnet, apply_ip_categorization


def assign_labels_by_day(df):
    day_to_label = {
        'mon': 'HTTPDoS',
        'tue': 'DDos',
        'thu': 'SSHBruteForce',
        'sun': 'Infiltration',
        'sat': 'Normal',  
        'wed': 'Normal'   
    }
    df['Label'] = df.apply(lambda row: day_to_label.get(row['day'], 'Normal') if row['Label'] == 'Attack' else row['Label'], axis=1)
    df = df[df['Label'] != 'Attack']
    return df

def extract_day_from_filename(filename):
    if "Mon" in filename:
        return "mon"
    elif "Tue" in filename:
        return "tue"
    elif "Thu" in filename:
        return "thu"
    elif "Sun" in filename:
        return "sun"
    else:
        return "other"

def merge_csv_files(input_dir):
    files = [
        "TestbedSatJun12Flows.csv",
        "TestbedSunJun13Flows.csv",
        "TestbedMonJun14Flows.csv",
        "TestbedTueJun15Flows.csv",
        "TestbedWedJun16Flows.csv",
        "TestbedThuJun17Flows.csv"
    ]
    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(input_dir, file))
        day = extract_day_from_filename(file)
        df['day'] = day
        df_list.append(df)
    df_merged = pd.concat(df_list, ignore_index=True)
    return df_merged

def categorize_ports(df):
    def port_category(port):
        if 0 <= port <= 1023:
            return 'Well-Known Ports'
        elif 1024 <= port <= 49151:
            return 'Registered Ports'
        elif 49152 <= port <= 65535:
            return 'Dynamic/Private Ports'
        else:
            return 'Unknown'

    df['sourcePortCategory'] = df['sourcePort'].apply(port_category)
    return df

def create_directory(path):
    absolute_path = os.path.abspath(path)
    if not os.path.exists(absolute_path):
        print(f"Creating directory: {absolute_path}")
        os.makedirs(absolute_path)
    else:
        print(f"Directory already exists: {absolute_path}")

def generate_protocol_distribution_table(df):
    protocol_counts = df.groupby(['protocolName', 'Label']).size().reset_index(name='count')
    total_counts = protocol_counts.groupby('Label')['count'].transform('sum')
    protocol_counts['percentage'] = (protocol_counts['count'] / total_counts) * 100
    return protocol_counts[['protocolName', 'Label', 'count', 'percentage']]

def apply_ip_categorization(df, source_col='source', destination_col='destination'):
    df['source_category'] = df[source_col].apply(categorize_ip_by_subnet)
    df['destination_category'] = df[destination_col].apply(categorize_ip_by_subnet)
    return df


def exploratory_analysis(input_dir):
    plot_dir = "D:\Documents\MANUELA-AFI\TFM\Experimento\plots/exploratory"
    create_directory(plot_dir)
    
    print("Merging files...")
    df = merge_csv_files(input_dir)
    
    print("Assigning labels based on the day of the week...")
    df = assign_labels_by_day(df)
    
    print("Categorizing ports...")
    df = categorize_ports(df)
    
    print("Ensuring the 'startDateTime' column is in datetime format...")
    df['startDateTime'] = pd.to_datetime(df['startDateTime'], errors='coerce')
    
    print("Creating 'hour' and 'minute' columns...")
    df['hour'] = df['startDateTime'].dt.hour
    df['minute'] = df['startDateTime'].dt.minute
    
    print("Generating plots and tables...")

    # Filter data into normal and malicious traffic
    df_normal = df[df['Label'] == 'Normal']
    df_malicious = df[df['Label'] != 'Normal']

    # Generar el gráfico de distribución de direcciones para tráfico normal
    direction_counts_normal = df_normal.groupby(['direction']).size().reset_index(name='counts')
    total_counts_normal = direction_counts_normal['counts'].sum()
    direction_counts_normal['percentage'] = 100 * direction_counts_normal['counts'] / total_counts_normal

    plt.figure(figsize=(14, 7))
    sns.barplot(x='direction', y='percentage', data=direction_counts_normal)
    plt.title('Distribución de Direcciones en Tráfico Normal')
    plt.ylabel('Porcentaje (%)')
    plt.savefig(os.path.join(plot_dir, 'distribucion_direcciones_normal.png'))
    plt.close()
    print("Saved: distribucion_direcciones_normal.png")

    # Generar el gráfico de distribución de direcciones para tráfico malicioso, separado por cada tipo de ataque
    attack_labels = df_malicious['Label'].unique()

    for label in attack_labels:
        direction_counts_attack = df_malicious[df_malicious['Label'] == label].groupby(['direction']).size().reset_index(name='counts')
        total_counts_attack = direction_counts_attack['counts'].sum()
        direction_counts_attack['percentage'] = 100 * direction_counts_attack['counts'] / total_counts_attack

        plt.figure(figsize=(14, 7))
        sns.barplot(x='direction', y='percentage', data=direction_counts_attack)
        plt.title(f'Distribución de Direcciones en Tráfico Malicioso: {label}')
        plt.ylabel('Porcentaje (%)')
        plt.savefig(os.path.join(plot_dir, f'distribucion_direcciones_{label}.png'))
        plt.close()
        print(f"Saved: distribucion_direcciones_{label}.png")

    # Histograma de Puertos Fuente: Tráfico Normal vs Malicioso
    plt.figure(figsize=(12, 6))
    sns.histplot(df_normal['sourcePort'], bins=50, color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(df_malicious['sourcePort'], bins=50, color='red', label='Malicioso', kde=True, stat="density", alpha=0.7)
    plt.title('Distribución de Puertos Fuente: Tráfico Normal vs Malicioso', fontsize=16, fontweight='bold')
    plt.xlabel('Puerto Fuente')
    plt.ylabel('Densidad')
    plt.legend(title='Tipo de Tráfico')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'distribucion_puertos_fuente.png'))
    plt.close()
    print("Saved: distribucion_puertos_fuente.png")

    # Histograma de Puertos Destino: Tráfico Normal vs Malicioso
    plt.figure(figsize=(12, 6))
    sns.histplot(df_normal['destinationPort'], bins=50, color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(df_malicious['destinationPort'], bins=50, color='red', label='Malicioso', kde=True, stat="density", alpha=0.7)
    plt.title('Distribución de Puertos Destino: Tráfico Normal vs Malicioso', fontsize=16, fontweight='bold')
    plt.xlabel('Puerto Destino')
    plt.ylabel('Densidad')
    plt.legend(title='Tipo de Tráfico')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'distribucion_puertos_destino.png'))
    plt.close()
    print("Saved: distribucion_puertos_destino.png")

    # Gráfico adicional: Histograma de Puertos Fuente (1-1024)
    plt.figure(figsize=(12, 6))
    sns.histplot(df_normal[df_normal['sourcePort'] <= 1024]['sourcePort'], bins=50, color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(df_malicious[df_malicious['sourcePort'] <= 1024]['sourcePort'], bins=50, color='red', label='Malicioso', kde=True, stat="density", alpha=0.7)
    plt.title('Distribución de Puertos Fuente (1-1024): Tráfico Normal vs Malicioso', fontsize=16, fontweight='bold')
    plt.xlabel('Puerto Fuente')
    plt.ylabel('Densidad')
    plt.legend(title='Tipo de Tráfico')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'distribucion_puertos_fuente_1024.png'))
    plt.close()
    print("Saved: distribucion_puertos_fuente_1024.png")

    # Gráfico adicional: Histograma de Puertos Destino (1-1024)
    plt.figure(figsize=(12, 6))
    sns.histplot(df_normal[df_normal['destinationPort'] <= 1024]['destinationPort'], bins=50, color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(df_malicious[df_malicious['destinationPort'] <= 1024]['destinationPort'], bins=50, color='red', label='Malicioso', kde=True, stat="density", alpha=0.7)
    plt.title('Distribución de Puertos Destino (1-1024): Tráfico Normal vs Malicioso', fontsize=16, fontweight='bold')
    plt.xlabel('Puerto Destino')
    plt.ylabel('Densidad')
    plt.legend(title='Tipo de Tráfico')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'distribucion_puertos_destino_1024.png'))
    plt.close()
    print("Saved: distribucion_puertos_destino_1024.png")

    # Evolución del Tráfico Normal y Malicioso con líneas de tiempo
    attack_labels = df_malicious['Label'].unique()
    colors = sns.color_palette('tab10', len(attack_labels))

    # Crear una columna 'Time' que combina hora y minutos para facilitar el plotting
    df_normal['Time'] = df_normal['hour'] + df_normal['minute'] / 60
    df_malicious['Time'] = df_malicious['hour'] + df_malicious['minute'] / 60

    plt.figure(figsize=(14, 20))  # Ajustar el tamaño de la figura para acomodar múltiples gráficos

    # Gráfico de líneas para tráfico normal (color azul)
    plt.subplot(len(attack_labels) + 1, 1, 1)  # Primer gráfico para tráfico normal
    sns.lineplot(data=df_normal, x='Time', y='totalSourceBytes', color='blue', linestyle='-', label='Normal')
    plt.title('Evolución del Tráfico Normal', fontsize=16, fontweight='bold')
    plt.xlabel('Hora del Día')
    plt.ylabel('Bytes')
    plt.yscale('log')  # Usar escala logarítmica para el eje Y
    plt.xlim(0, 24)  # Limitar el eje X de 0 a 24 horas
    plt.xticks(ticks=range(0, 25, 1), labels=[f'{int(t)}:00' for t in range(0, 25, 1)])
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Gráfico de líneas individual para cada tipo de ataque en tráfico malicioso, superponiendo tráfico normal
    for i, (label, color) in enumerate(zip(attack_labels, colors), start=2):
        plt.subplot(len(attack_labels) + 1, 1, i)  # Crear un subplot para cada ataque
        df_malicious_label = df_malicious[df_malicious['Label'] == label]
        sns.lineplot(data=df_malicious_label, x='Time', y='totalSourceBytes', color=color, linestyle='--', label=label)
        sns.lineplot(data=df_normal, x='Time', y='totalSourceBytes', color='blue', linestyle='-', alpha=0.3, label='Normal')  # Superponer tráfico normal con transparencia
        plt.title(f'Evolución del Tráfico Malicioso: {label}', fontsize=16, fontweight='bold')
        plt.xlabel('Hora del Día')
        plt.ylabel('Bytes')
        plt.yscale('log')  # Usar escala logarítmica para el eje Y
        plt.xlim(0, 24)  # Limitar el eje X de 0 a 24 horas
        plt.xticks(ticks=range(0, 25, 1), labels=[f'{int(t)}:00' for t in range(0, 25, 1)])
        plt.legend(loc='upper right')
        plt.tight_layout()

    # Ajustar el layout general
    plt.tight_layout()

    # Guardar el gráfico de evolución de tráfico
    plt.savefig(os.path.join(plot_dir, 'evolucion_trafico_malicioso.png'))
    plt.close()
    print("Saved: evolucion_trafico_malicioso.png")

    # Grafo de conexiones de IPs en tráfico malicioso
    for attack_type in attack_labels:
        attack_data = df_malicious[df_malicious['Label'] == attack_type]
        G = nx.from_pandas_edgelist(attack_data, 'source', 'destination', create_using=nx.DiGraph())

        # Grafo sin etiquetas de IP
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='red')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='blue')
        plt.title(f'Grafo de Conexiones de IPs: {attack_type}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(plot_dir, f'grafos_conexiones_{attack_type}.png'))
        plt.close()
        print(f"Saved: grafos_conexiones_{attack_type}.png")

        # Grafo con etiquetas de IPs
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='red')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='blue')
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
        plt.title(f'Grafo de Conexiones de IPs con Etiquetas: {attack_type}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(plot_dir, f'grafos_conexiones_etiquetas_{attack_type}.png'))
        plt.close()
        print(f"Saved: grafos_conexiones_etiquetas_{attack_type}.png")


    print("Applying IP categorization...")
    df = apply_ip_categorization(df)
    # Filter data into normal and malicious traffic
    df_normal = df[df['Label'] == 'Normal']
    df_malicious = df[df['Label'] != 'Normal']

    # Grafo de conexiones de IPs en tráfico malicioso con color basado en categorías de IP
    color_map = {
        '192.168.1.0/24': 'blue',
        '192.168.2.0/24': 'green',
        '192.168.3.0/24': 'orange',
        '192.168.4.0/24': 'purple',
        '192.168.5.0/24': 'red',
        '192.168.6.0/24': 'brown',
        'loopback': 'pink',
        'multicast': 'yellow',
        'private (other)': 'gray',
        'link-local': 'cyan',
        'carrier-grade NAT': 'magenta',
        'documentation': 'lime',
        'External': 'black'
    }

    legend_labels = list(color_map.keys())
    legend_colors = list(color_map.values())

    for attack_type in df_malicious['Label'].unique():
        attack_data = df_malicious[df_malicious['Label'] == attack_type]
        G = nx.from_pandas_edgelist(attack_data, 'source', 'destination', create_using=nx.DiGraph())

        # Node colors based on IP categorization
        node_colors = [color_map[categorize_ip_by_subnet(node)] for node in G.nodes()]

        # Grafo sin etiquetas de IP
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='blue')

         # Add legend
        for label, color in zip(legend_labels, legend_colors):
            plt.scatter([], [], color=color, label=label)
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='best')

        plt.title(f'Grafo de Conexiones de IPs: {attack_type}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(plot_dir, f'grafos_conexiones_{attack_type}.png'))
        plt.close()
        print(f"Saved: grafos_conexiones_{attack_type}.png")

        # Grafo con etiquetas de IPs
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='blue')
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        # Add legend
        for label, color in zip(legend_labels, legend_colors):
            plt.scatter([], [], color=color, label=label)
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='best')

        plt.title(f'Grafo de Conexiones de IPs con Etiquetas: {attack_type}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(plot_dir, f'grafos_conexiones_etiquetas_{attack_type}.png'))
        plt.close()
        print(f"Saved: grafos_conexiones_etiquetas_{attack_type}.png")


if __name__ == "__main__":
    exploratory_analysis("data/raw/")
