# app1

Frontend Flutter para la aplicación móvil del TFG de reconocimiento de lengua de signos a texto.

## Alcance actual

Esta carpeta contiene una primera versión del frontend, sin backend ni inferencia real. El objetivo es construir una base visual y estructural seria sobre la que más adelante se puedan conectar:

- captura de cámara
- streaming de frames o vídeo
- traducción en directo
- revisión y guardado de transcripciones
- historial de sesiones

## Dirección de diseño

La interfaz sigue una línea visual sobria y profesional:

- inspiración institucional y premium
- paleta marfil, azul marino y dorado contenido
- jerarquía tipográfica clara
- componentes amplios y espaciados
- modo demo con datos mock

## Pantallas incluidas

- `Inicio`
- `Directo`
- `Historial`
- `Ajustes`

## Estructura

```text
lib/
├── app.dart
├── core/
│   ├── navigation/
│   ├── theme/
│   └── widgets/
├── features/
│   ├── history/
│   ├── home/
│   ├── live_translation/
│   └── settings/
└── shared/
    └── data/
```

## Ejecutar

```bash
flutter pub get
flutter run
```

## Próximos pasos naturales

- integrar permisos y preview real de cámara
- añadir flujo de revisión detallada de transcripción
- conectar capa de servicios y backend
- introducir persistencia local para historial y borradores
