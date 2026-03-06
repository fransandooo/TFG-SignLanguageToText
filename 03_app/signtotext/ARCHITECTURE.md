# SignToText - Estructura del Proyecto

## 📁 Organización del Código

El proyecto está organizado de manera modular para facilitar el mantenimiento y escalabilidad:

```
lib/
├── main.dart                           # Punto de entrada de la aplicación
├── theme_provider.dart                 # Gestión de temas (claro/oscuro)
├── app_fonts.dart                      # Helper para fuentes con fallback seguro
│
├── models/                             # Modelos de datos
│   └── bounding_box.dart              # Punto y BoundingBox
│
├── screens/                            # Pantallas principales
│   ├── main_screen.dart               # Scaffold principal con AppBar
│   ├── home_screen.dart               # Pantalla principal con lógica
│   ├── home_header.dart               # Header de la pantalla principal
│   └── process_button.dart            # Botón de procesamiento
│
├── widgets/                            # Componentes reutilizables
│   ├── image_preview_card.dart        # Card de preview de imagen
│   ├── bounding_box_input_card.dart   # Card de input de coordenadas
│   ├── processing_result_card.dart    # Card de resultado del procesamiento
│   └── image_source_bottom_sheet.dart # Bottom sheet para selección de fuente
│
├── services/                           # Lógica de negocio
│   └── image_processing_service.dart  # Servicio de procesamiento de imágenes
│
└── utils/                              # Utilidades
    └── snackbar_utils.dart            # Utilidades para notificaciones
```

## 🎯 Responsabilidades

### Models

- **bounding_box.dart**: Define las estructuras de datos `Point` y `BoundingBox`

### Screens

- **main_screen.dart**: Contiene el Scaffold con AppBar y botón de cambio de tema
- **home_screen.dart**: Lógica principal de la aplicación y estado
- **home_header.dart**: Título y descripción de la pantalla
- **process_button.dart**: Botón de procesamiento con animaciones

### Widgets

- Componentes UI reutilizables y autocontenidos
- Cada widget es responsable de su propia presentación
- Reciben datos y callbacks como parámetros

### Services

- **image_processing_service.dart**: Encapsula toda la lógica de procesamiento de imágenes

### Utils

- **snackbar_utils.dart**: Métodos helper para mostrar notificaciones consistentes

## 🔧 Ventajas de esta Estructura

1. **Modularidad**: Cada archivo tiene una responsabilidad clara
2. **Mantenibilidad**: Fácil localizar y modificar funcionalidades
3. **Reutilización**: Widgets y servicios pueden usarse en otras partes
4. **Testeable**: Cada componente puede probarse de forma aislada
5. **Escalable**: Fácil agregar nuevas funcionalidades
6. **Legible**: Código organizado y fácil de entender

## 🚀 Próximos Pasos

Para extender la funcionalidad:

1. Agregar nuevas pantallas en `screens/`
2. Crear nuevos widgets reutilizables en `widgets/`
3. Implementar servicios adicionales en `services/`
4. Definir nuevos modelos en `models/`
