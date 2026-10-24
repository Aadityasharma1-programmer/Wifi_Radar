/*
 * Wi-Fi Radar - ESP32 CSI collector (ESP-IDF 5.x, classic ESP32).
 *
 * The ESP32 joins your router as a station and pings the gateway ~50x/s.
 * Every received frame produces CSI, printed over USB serial as:
 *     CSI_DATA,<seq>,<rssi>,<len>,[i q i q ...]
 * (first byte of each pair = imaginary, second = real; 64 subcarriers with LLTF).
 *
 * Put the ESP32 and the router on opposite sides of the area you want to monitor so a
 * person walking in between disturbs the link.
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "lwip/ip_addr.h"
#include "ping/ping_sock.h"

#define WIFI_SSID        "YOUR_WIFI_SSID"
#define WIFI_PASS        "YOUR_WIFI_PASSWORD"
#define PING_INTERVAL_MS 20            /* 20 ms -> ~50 packets/s */

#define GOT_IP_BIT BIT0
static EventGroupHandle_t s_events;
static esp_netif_t *s_netif;
static uint32_t s_seq;

static void csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) return;
    printf("CSI_DATA,%lu,%d,%d,[", (unsigned long)s_seq++, info->rx_ctrl.rssi, info->len);
    for (int i = 0; i < info->len; i++) {
        printf("%d%s", info->buf[i], (i == info->len - 1) ? "" : " ");
    }
    printf("]\n");
}

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && (id == WIFI_EVENT_STA_START || id == WIFI_EVENT_STA_DISCONNECTED)) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_events, GOT_IP_BIT);
    }
}

static void start_ping_gateway(void)
{
    esp_netif_ip_info_t ip;
    esp_netif_get_ip_info(s_netif, &ip);

    ip_addr_t target = {0};
    target.type = IPADDR_TYPE_V4;
    target.u_addr.ip4.addr = ip.gw.addr;

    esp_ping_config_t cfg = ESP_PING_DEFAULT_CONFIG();
    cfg.target_addr = target;
    cfg.count = ESP_PING_COUNT_INFINITE;
    cfg.interval_ms = PING_INTERVAL_MS;
    cfg.timeout_ms = 500;
    cfg.data_size = 1;

    esp_ping_callbacks_t cbs = {0};
    esp_ping_handle_t h;
    ESP_ERROR_CHECK(esp_ping_new_session(&cfg, &cbs, &h));
    ESP_ERROR_CHECK(esp_ping_start(h));
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_events = xEventGroupCreate();
    s_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL));

    wifi_config_t wc = {0};
    strncpy((char *)wc.sta.ssid, WIFI_SSID, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, WIFI_PASS, sizeof(wc.sta.password) - 1);
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE);                       /* no power-save: steady packet rate */

    xEventGroupWaitBits(s_events, GOT_IP_BIT, pdFALSE, pdTRUE, portMAX_DELAY);

    wifi_csi_config_t csi = {
        .lltf_en = true,            /* legacy LTF -> 64 subcarriers (128 bytes) */
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    start_ping_gateway();
}
