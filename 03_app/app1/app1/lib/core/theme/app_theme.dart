import 'package:flutter/material.dart';

import 'app_colors.dart';

abstract final class AppTheme {
  static ThemeData build() {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: AppColors.ivory,
      colorScheme: const ColorScheme.light(
        primary: AppColors.midnight,
        onPrimary: AppColors.ivory,
        secondary: AppColors.gold,
        onSecondary: AppColors.ink,
        surface: AppColors.parchment,
        onSurface: AppColors.ink,
        error: AppColors.crimson,
        onError: AppColors.ivory,
      ),
    );

    const display = 'Georgia';
    const body = 'Arial';

    return base.copyWith(
      textTheme: base.textTheme.copyWith(
        displaySmall: const TextStyle(
          fontFamily: display,
          fontSize: 34,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.8,
          height: 1.1,
          color: AppColors.ink,
        ),
        headlineMedium: const TextStyle(
          fontFamily: display,
          fontSize: 24,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.4,
          height: 1.15,
          color: AppColors.ink,
        ),
        titleLarge: const TextStyle(
          fontFamily: body,
          fontSize: 20,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
          color: AppColors.ink,
        ),
        titleMedium: const TextStyle(
          fontFamily: body,
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: AppColors.ink,
        ),
        bodyLarge: const TextStyle(
          fontFamily: body,
          fontSize: 16,
          height: 1.5,
          color: AppColors.ink,
        ),
        bodyMedium: const TextStyle(
          fontFamily: body,
          fontSize: 14,
          height: 1.5,
          color: AppColors.steel,
        ),
        labelLarge: const TextStyle(
          fontFamily: body,
          fontSize: 14,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.3,
        ),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        foregroundColor: AppColors.ink,
        elevation: 0,
        centerTitle: false,
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: AppColors.parchment,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(28),
          side: const BorderSide(color: AppColors.line),
        ),
      ),
      navigationBarTheme: NavigationBarThemeData(
        indicatorColor: AppColors.midnight,
        backgroundColor: AppColors.ivory,
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          return TextStyle(
            fontFamily: body,
            fontSize: 12,
            fontWeight: states.contains(WidgetState.selected)
                ? FontWeight.w700
                : FontWeight.w500,
            color: states.contains(WidgetState.selected)
                ? AppColors.ivory
                : AppColors.steel,
          );
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          return IconThemeData(
            color: states.contains(WidgetState.selected)
                ? AppColors.ivory
                : AppColors.steel,
          );
        }),
      ),
      chipTheme: base.chipTheme.copyWith(
        backgroundColor: AppColors.ivory,
        side: const BorderSide(color: AppColors.line),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
      ),
      dividerTheme: const DividerThemeData(color: AppColors.line, thickness: 1),
    );
  }
}
