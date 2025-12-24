import krpc
import time

# -------------------------------
# Профиль тангажа (высота, угол)
# -------------------------------
# Тангаж: 90° — строго вертикально, 0° — горизонтально.
PITCH_PROFILE = [
    (0,      90),  # старт: строго вверх
    (500,    90),  # первые 500 м держим вертикаль
    (2000,   80),  # к 2 км начинаем слегка наклоняться
    (5000,   70),  # 5 км
    (10000,  60),  # 10 км
    (30000,  40),  # 30 км
    (45000,  20),  # 45 км
    (60000,   0),  # 60 км
]

TARGET_ORBIT_ALT = 70000.0


def interpolate_pitch(altitude: float) -> float:
    """
    Линейная интерполяция: по текущей высоте возвращает целевой тангаж.
    """
    # Ниже первой точки — берём её значение
    if altitude <= PITCH_PROFILE[0][0]:
        return PITCH_PROFILE[0][1]

    # Выше последней точки — берём значение последней
    if altitude >= PITCH_PROFILE[-1][0]:
        return PITCH_PROFILE[-1][1]

    # Ищем интервал, в который попадает текущая высота
    for i in range(len(PITCH_PROFILE) - 1):
        h0, pitch0 = PITCH_PROFILE[i]
        h1, pitch1 = PITCH_PROFILE[i + 1]

        if h0 <= altitude <= h1:
            # Линейная интерполяция между (h0, pitch0) и (h1, pitch1)
            t = (altitude - h0) / (h1 - h0)  # от 0 до 1
            return pitch0 + (pitch1 - pitch0) * t
    return PITCH_PROFILE[-1][1]


def main():
    conn = krpc.connect(name='Autopilot')
    sc = conn.space_center
    vessel = sc.active_vessel

    # Берём полёт относительно поверхности
    flight = vessel.flight(vessel.surface_reference_frame)

    ap = vessel.auto_pilot
    ap.reference_frame = vessel.surface_reference_frame

    # Начальная ориентация — строго вверх, на восток
    ap.target_pitch_and_heading(90, 90)
    ap.engage()

    vessel.control.sas = False
    vessel.control.throttle = 1.0

    print('Запуск')
    time.sleep(1)
    vessel.control.activate_next_stage()  # Старт двигателей

    while True:
        altitude = flight.surface_altitude  # высота над поверхностью

        if altitude >= TARGET_ORBIT_ALT:
            print('Достигнута высота 70 км, автопилот завершает работу.')
            break

        # Вычисляем целевой тангаж по высоте (линейная интерполяция)
        target_pitch = interpolate_pitch(altitude)
        target_heading = 90.0  # курс на восток

        # Задаём ориентацию автопилоту
        ap.target_pitch_and_heading(target_pitch, target_heading)

        time.sleep(0.1)

    ap.disengage()
    print('Автопилот выключен на высоте ~70 км.')


if __name__ == '__main__':
    main()
