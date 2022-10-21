ENABLE_ALL_FEATURES = False


if ENABLE_ALL_FEATURES:
    FEATURE_NAMES = [
        'latency', 'cpu_use', 'mem_use_percent', 'mem_use_amount',
        'file_write_rate', 'file_read_rate',
        'net_send_rate', 'net_receive_rate', 'http_status'
    ]
else:
    FEATURE_NAMES = [
        'latency', 'http_status'
    ]

FAULT_TYPES = {'delay', 'abort', 'cpu'}

INVOLVED_SERVICES = [
    'route-plan',
    'food',
    'config',
    'order',
    'seat',
    'train',
    'travel-plan',
    'user',
    'route',
    'ticketinfo',
    'verification-code',
    'price',
    'contacts',
    'cancel',
    'travel2',
    'assurance',
    'preserve',
    'basic',
    'auth',
    'security',
    'consign',
    'food-map',
    'travel',
    'station',
    'ui-dashboard',
    'preserve-other',
    'order-other',
    'inside-payment',
    'execute',
    'payment',
    'admin-order',
    'admin-basic-info',
    'gateway',
    'admin-route',
    'admin-travel',
    'notification',
    'admin-user',
    'istio-mixer',
    'kibana',
    'jaeger-query',
]


SERVICE2IDX = {service: idx for idx, service in enumerate(INVOLVED_SERVICES)}

EXP_NOISE_LIST = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
# EXP_NOISE_LIST = [0]

SIMPLE_NAME_DICT = {
    'ts-route-plan-service':'route-plan',
    'ts-food-service':'food',
    'ts-config-service':'config',
    'ts-order-service':'order',
    'ts-seat-service':'seat',
    'ts-train-service':'train',
    'ts-travel-plan-service':'travel-plan',
    'ts-user-service':'user',
    'ts-route-service':'route',
    'ts-ticketinfo-service':'ticketinfo',
    'ts-verification-code-service':'verification-code',
    'ts-price-service':'price',
    'ts-contacts-service':'contacts',
    'ts-cancel-service':'cancel',
    'ts-travel2-service':'travel2',
    'ts-assurance-service':'assurance',
    'ts-preserve-service':'preserve',
    'ts-basic-service':'basic',
    'ts-auth-service':'auth',
    'ts-security-service':'security',
    'ts-consign-service':'consign',
    'ts-food-map-service':'food-map',
    'ts-travel-service':'travel',
    'ts-station-service':'station',
    'ts-ui-dashboard-service':'ui-dashboard',
    'ts-preserve-other-service':'preserve-other',
    'ts-order-other-service':'order-other',
    'ts-inside-payment-service':'inside-payment',
    'ts-execute-service':'execute',
    'ts-payment-service':'payment',
    'ts-admin-order-service':'admin-order',
    'ts-admin-basic-info-service':'admin-basic-info',
    'istio-ingressgateway':'gateway',
    'ts-admin-route-service':'admin-route',
    'ts-admin-travel-service':'admin-travel',
    'ts-notification-service':'notification',
    'ts-admin-user-service':'admin-user',
    'ts-istio-mixer-service':'istio-mixer',
    'ts-kibana-service':'kibana',
    'ts-jaeger-query-service':'jaeger-query',
}
