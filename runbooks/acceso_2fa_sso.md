# Runbook: Autenticación de dos factores (2FA) y SSO

**Categoría:** acceso  
**Aplica a:** Todos los productos con autenticación corporativa

---

## Problemas con código de verificación 2FA

### El código llega pero expira antes de usarlo

Los códigos TOTP tienen una ventana de 30 segundos. Si el usuario los introduce tarde:
1. Verificar que el reloj del dispositivo esté sincronizado (NTP). Una diferencia de más de 30s invalida los códigos.
2. En Android/iOS: Ajustes → Fecha y hora → "Usar hora de red".
3. Como solución temporal, el administrador puede ampliar la ventana de tolerancia a 90s desde el panel: Seguridad → 2FA → Tolerancia de tiempo.
4. Si el problema persiste, regenerar el secreto TOTP del usuario: Administración → Usuario → Restablecer 2FA.

### El usuario no recibe el SMS/email con el código

1. Verificar que el número de teléfono o email secundario registrado sea correcto.
2. Revisar carpeta de spam para el email.
3. Para SMS: confirmar que el número tiene prefijo internacional correcto.
4. Si el canal está bloqueado, usar el código de recuperación que se generó al activar el 2FA.
5. Si perdió los códigos de recuperación: escalar a Nivel 2 con verificación de identidad por video.

### Deshabilitar 2FA temporalmente (emergencia)

Solo con autorización del responsable del tenant:
1. Panel de administración → Seguridad → Políticas de autenticación.
2. Desactivar "2FA obligatorio" temporalmente (máximo 4 horas).
3. Registrar la acción en el log de cambios con justificación.

---

## Problemas con Single Sign-On (SSO)

### SSO dejó de funcionar tras actualización del navegador

Específicamente reportado en Chrome después de actualizaciones mayores:
1. Pedir al usuario que limpie cookies y caché del sitio (no del navegador completo): F12 → Application → Cookies → Clear.
2. Si usa contraseñas guardadas en Chrome: Configuración → Contraseñas → eliminar la entrada de la aplicación y volver a guardar.
3. Verificar que las extensiones de seguridad corporativas (ej. proxies, antivirus) no estén bloqueando la redirección SSO.
4. Si el problema afecta a toda la organización, verificar la configuración del IdP (Identity Provider) — puede requerir actualizar los metadatos SAML.

### El usuario ve la pantalla de login en lugar del SSO corporativo

Significa que el SSO no está correctamente configurado para ese usuario:
1. Verificar que el dominio del email del usuario coincide con el dominio SSO registrado en el tenant.
2. Si el usuario cambió de departamento, puede necesitar ser reasignado al grupo SSO correcto.
3. Revisar si el atributo `department` en el directorio corporativo está actualizado.

---

## Escalado

Escalar a Nivel 2 si:
- El IdP (Active Directory, Okta, Azure AD) no responde.
- El certificado SSL del IdP está vencido.
- Más del 20% de usuarios de un tenant no pueden autenticarse.
